import io

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen

from app.screens.services.image_library_service import ImageLibraryService, sha256_hex
from app.screens.utils.additional import BaseScreen, ImageMDButton, SelectableImage
from app.screens.utils.custom_logging import get_logger
from app.screens.utils.utils import extend_key

logger = get_logger(__name__)


class DbViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_service = ImageLibraryService(db=self.db)
        self.grid_1 = None
        self.grid_2 = None
        self.key = ""
        self.loaded = False
        self.selected_images = []
        self.prev_line_color = None
        self.checkbox_first = None
        self.last_match = {}

        self.autoload_on_enter = True
        self._autoload_ev = None

    def on_enter(self, *args):
        self.setup_header()
        self.key = extend_key(self.manager.get_screen("login").key)
        self.grid_1 = self.ids.grid_1
        self.grid_2 = self.ids.grid_2

        if self._autoload_ev is not None:
            self._autoload_ev.cancel()
            self._autoload_ev = None

        if self.autoload_on_enter:
            self._autoload_ev = Clock.schedule_once(lambda dt: self.show_db_images(), 0)

    def show_db_images(self):
        self.grid_1.clear_widgets()
        self.grid_2.clear_widgets()
        self.unselect_all_images()

        self.last_match = {
            "matched_simple": set(),
            "matched_secure": set(),
            "simple": 0,
            "secure": 0,
        }

        plain_images, secure_images = self.image_service.load_and_decode_db_images(self.key)
        total_count = len(plain_images) + len(secure_images)

        if total_count == 0:
            self.toggle_load_label("on", text="No images in DB.")
            self.loaded = False
            return
        else:
            self.toggle_load_label("on")
            self.loaded = True

        simple_count = self._render_image_list(
            plain_images, self.grid_1, (1.0, 0.6, 0.0, 0.5), "matched_simple"
        )
        secure_count = self._render_image_list(
            secure_images, self.grid_2, (0.0, 1.0, 0.0, 0.5), "matched_secure"
        )

        self.last_match["simple"] = simple_count
        self.last_match["secure"] = secure_count

        self.update_label_info(simple_count, secure_count)
        self.toggle_load_label("off")

    def _render_image_list(
        self,
        records: list[tuple[int, bytes, str]],
        grid,
        line_color: tuple[float, float, float, float],
        match_key: str,
    ) -> int:
        count = 0
        for pk, b_image, ext in records:
            selectable_img = SelectableImage()
            try:
                data = io.BytesIO(b_image)
                texture = CoreImage(data, ext=ext).texture
                selectable_img.line_color = line_color
                count += 1
                self.last_match[match_key].add(sha256_hex(b_image))
            except Exception as e:
                logger.warning(f"Failed to display image {pk}: {e}")
                texture = self._create_error_texture()
                selectable_img.line_color = (1.0, 0.0, 0.0, 0.5)

            selectable_img.source = str(pk)
            selectable_img.texture = texture
            selectable_img.ids.img.bind(on_press=self.image_click)
            selectable_img.ids.checkbox.bind(on_press=self.checkbox_click)
            grid.add_widget(selectable_img)
        return count

    def _create_error_texture(self) -> Texture:
        img = np.zeros((600, 800, 1), dtype=np.float32)
        h, w, *_ = img.shape
        red = (255, 0, 0)
        img = cv2.line(img, (0, 0), (w, h), red, thickness=6)
        img = cv2.line(img, (w, 0), (0, h), red, thickness=6)
        buff = bytes(img.flatten())
        texture = Texture.create(size=(w, h))
        texture.blit_buffer(buff, bufferfmt="ubyte", colorfmt="bgr")
        return texture

    def update_label_info(self, simple, secure):
        self.ids.simple.text = f"Simple images [{simple}]"
        self.ids.secure.text = f"Secure images [{secure}]"

    def checkbox_click(self, instance):
        logger.debug(f"{instance}")
        self.checkbox_first = True
        self.update_buttons_state()

    def image_click(self, instance):
        if instance in self.selected_images:
            self.unselect_image(instance)
            return

        if not self.checkbox_first:
            self.unselect_all_images()
        else:
            self.checkbox_first = False

        self.select_image(instance)

    def select_image(self, instance):
        self.selected_images.append(instance)
        instance.parent.selected = True
        self.update_buttons_state()

    def unselect_image(self, instance):
        if instance in self.selected_images:
            instance.parent.selected = False
            self.selected_images.remove(instance)
        self.update_buttons_state()

    def unselect_all_images(self):
        images = self.selected_images.copy()
        for image in images:
            self.unselect_image(image)
        self.update_buttons_state()

    def preview_img(self):
        if len(self.selected_images) != 1:
            return

        popup = Popup(title="Preview", size_hint=(None, None), size=(700, 500))

        img = ImageMDButton()
        img.texture = self.selected_images[0].texture
        img.bind(on_press=lambda x: popup.dismiss())

        popup.content = img
        popup.open()

    def delete_from_db(self):
        if len(self.selected_images) < 1:
            return

        for image in self.selected_images:
            key = int(image.source)
            self.image_service.delete_db_image(key)

        self.unselect_all_images()
        self.show_db_images()

    def update_buttons_state(self):
        if len(self.selected_images) > 0:
            self.ids.delete.disabled = False
        else:
            self.ids.delete.disabled = True

        if len(self.selected_images) == 1:
            self.ids.preview.disabled = False
        else:
            self.ids.preview.disabled = True
