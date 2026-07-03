import hashlib

from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen

from app.screens.utils.additional import BaseScreen, ImageMDButton, SelectableImage
from app.screens.utils.custom_logging import get_logger
from app.screens.utils.utils import extend_key

logger = get_logger(__name__)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def guess_image_ext(data: bytes) -> str | None:
    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    # JPEG signature: FF D8 FF
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"  # Kivy loaders usually treat jpg/jpeg as "jpg"

    # GIF
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"

    # BMP
    if data.startswith(b"BM"):
        return "bmp"

    return "png"  # default to png


class DbViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_1 = None
        self.grid_2 = None
        self.key = ""
        self.loaded = False
        self.selected_images = []
        self.prev_line_color = None
        self.checkbox_first = None
        self.last_match = dict()

        self.autoload_on_enter = True
        self._autoload_ev = None

    def on_enter(self, *args):
        self.setup_header()
        self.key = extend_key(self.manager.get_screen("login").key)
        self.grid_1 = self.ids.grid_1
        self.grid_2 = self.ids.grid_2
        # TODO: update property and add smth like hash check to reload if db images updated
        #       and probably reload only updated grid, but not all images
        # if not self.loaded:
        if self._autoload_ev is not None:
            self._autoload_ev.cancel()
            self._autoload_ev = None

        if self.autoload_on_enter:
            self._autoload_ev = Clock.schedule_once(lambda dt: self.show_db_images(), 0)

    def show_db_images(self):
        import io

        import cv2
        import numpy as np
        from Cryptodome.Cipher import AES

        db_images = self.db.get_images()
        self.grid_1.clear_widgets()
        self.grid_2.clear_widgets()
        self.unselect_all_images()

        self.last_match = {
            "matched_simple": set(),
            "matched_secure": set(),
            "simple": 0,
            "secure": 0,
        }

        if len(db_images) == 0:
            self.toggle_load_label("on", text="No images in DB.")
            self.loaded = False
            return
        else:
            self.toggle_load_label("on")
            self.loaded = True

        simple, secure = 0, 0
        for pk, b_image in db_images:
            selectable_img = SelectableImage()

            success = False
            grid, texture = None, None
            try:
                ext = guess_image_ext(b_image)
                data = io.BytesIO(b_image)
                texture = CoreImage(data, ext=ext).texture
                success = True
                selectable_img.line_color = (1.0, 0.6, 0.0, 0.5)
                grid = self.grid_1
                simple += 1
                self.last_match["matched_simple"].add(sha256(b_image))

            except Exception as e:
                logger.warning(f"fail to load {e}")

            if not success:  # try to decrypt
                try:
                    nonce = b_image[:16]
                    tag = b_image[16:32]
                    ciphertext = b_image[32:]

                    cipher = AES.new(self.key, AES.MODE_EAX, nonce=nonce)
                    plain = cipher.decrypt_and_verify(ciphertext, tag)

                    ext = guess_image_ext(plain)
                    data = io.BytesIO(plain)
                    texture = CoreImage(data, ext=ext).texture
                    success = True
                    selectable_img.line_color = (0.0, 1.0, 0.0, 0.5)
                    grid = self.grid_2
                    secure += 1
                    self.last_match["matched_secure"].add(sha256(plain))

                except Exception as e:
                    logger.warning(f"fail to decrypt {e}")

            if not success:  # show cross instead of image
                img = np.zeros((600, 800, 1), dtype=np.float32)  # make multiple crosses
                # img = np.zeros((600, 800, 3), dtype=np.float32)
                h, w, *_ = img.shape
                red = (255, 0, 0)
                img = cv2.line(img, (0, 0), (w, h), red, thickness=6)
                img = cv2.line(img, (w, 0), (0, h), red, thickness=6)
                buff = bytes(img.flatten())

                texture = Texture.create(size=(w, h))
                texture.blit_buffer(buff, bufferfmt="ubyte", colorfmt="bgr")
                selectable_img.line_color = (1.0, 0.0, 0.0, 0.5)
                grid = self.grid_1

            selectable_img.source = str(pk)
            selectable_img.texture = texture
            selectable_img.ids.img.bind(on_press=self.image_click)
            selectable_img.ids.checkbox.bind(on_press=self.checkbox_click)

            grid.add_widget(selectable_img)

        self.last_match["simple"] = simple
        self.last_match["secure"] = secure

        self.update_label_info(simple, secure)
        self.toggle_load_label("off")

    def update_label_info(self, simple, secure):
        self.ids.simple.text = f"Simple images [{simple}]"
        self.ids.secure.text = f"Secure images [{secure}]"

    def checkbox_click(self, instance):
        logger.debug(f"{instance}")
        self.checkbox_first = True
        self.update_buttons_state()

    def image_click(self, instance):
        # path = instance.source

        if instance in self.selected_images:
            self.unselect_image(instance)
            return

        if not self.checkbox_first:
            self.unselect_all_images()  # remove all
        else:
            self.checkbox_first = False

        self.select_image(instance)  # choose new

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
            self.db.delete_image(key)

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
