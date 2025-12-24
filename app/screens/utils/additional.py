import os
from base64 import b64encode
from typing import Tuple

from kivy.clock import Clock
from kivy.uix.behaviors.button import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivymd.uix import SpecificBackgroundColorBehavior
from kivymd.uix.behaviors import HoverBehavior
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import ButtonBehavior as MDButtonBehavior
from kivymd.uix.label import MDLabel

from app.screens.utils.custom_logging import get_logger
from app.screens.utils.db import DB

logger = get_logger(__name__)


class MDLabelBtn(ButtonBehavior, MDLabel, HoverBehavior):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.allow_hover = False
        self.saved_color = None

    def on_enter(self):
        if self.allow_hover:
            self.saved_color = self.md_bg_color.copy()
            self.md_bg_color = (1, 1, 1, 0.1)

    def on_leave(self):
        if self.allow_hover:
            self.md_bg_color = self.saved_color


class ImageMDButton(
    MDButtonBehavior, Image, SpecificBackgroundColorBehavior, HoverBehavior
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.allow_hover = False
        self.saved_color = None

    def on_enter(self):
        if self.allow_hover:
            self.saved_color = self.md_bg_color.copy()
            self.md_bg_color = (1, 1, 1, 0.1)

    def on_leave(self):
        if self.allow_hover:
            self.md_bg_color = self.saved_color


class BaseScreen:
    manager = None
    progress_bar = None
    ids = None
    key = None

    def __init__(self):
        self.exit_screen = False
        self.db = DB()

    def label_out(self, text: str):
        """Put string message to the label"""
        self.ids.word_label.text = text

    def get_input(self) -> str:
        return self.ids.word_input.text

    def encrypt(self, text: str) -> str:
        from Cryptodome.Cipher import AES

        cipher = AES.new(self.key, AES.MODE_EAX, nonce=b"TODO")
        encoded_text = cipher.encrypt(text.encode("utf-8"))
        b_encoded_text = b64encode(encoded_text).decode("utf-8")
        return b_encoded_text

    def select_direction(self, screen_name: str):
        self.exit_screen = True
        translations = {
            "main": 0,
            "imageview": 1,
            "dbview": 2,
            "mlview": 3,
            "detectionview": 4,
            "settingsview": 5,
        }

        old = translations[self.manager.current]
        new = translations[screen_name]

        if old < new:
            self.manager.transition.direction = "left"
        else:
            self.manager.transition.direction = "right"

        self.manager.current = screen_name

    def toggle_load_label(self, mode: str):
        lbl: MDLabel = self.ids.load_label

        def lbl_prop(
            text: str = "",
            lbl_hint_y: float = 0.1,
            color: Tuple[float, float, float, float] = (1, 1, 1, 1),
            pbar_hint_y: float = 0.1,
            opacity: float = 1,
        ):
            lbl.text = text
            lbl.size_hint_y = lbl_hint_y
            lbl.color = color
            self.progress_bar.size_hint_y = pbar_hint_y
            self.progress_bar.opacity = opacity

        if mode == "on":
            lbl_prop("Loading, please wait...")

        elif mode == "no_dir":
            lbl_prop("No images, please select folder.", opacity=0)

        elif mode == "success":
            lbl_prop("Success!", color=(0, 1, 0, 1))
            Clock.schedule_once(lambda tm: self.toggle_load_label("off"), 1)

        elif mode == "off":
            lbl_prop(lbl_hint_y=0, pbar_hint_y=0, opacity=0)

    def goto_images(self):
        self.select_direction("imageview")

    def goto_main(self):
        self.select_direction("main")

    def goto_db(self):
        self.select_direction("dbview")

    def goto_ml(self):
        self.select_direction("mlview")

    def goto_detection(self):
        self.select_direction("detectionview")

    def goto_settings(self):
        self.select_direction("settingsview")


class Header(MDBoxLayout, BaseScreen):
    def __init__(self, **kwargs):  # TODO: check
        super().__init__(**kwargs)


class MlUiHelper:
    projects_folder = None
    main_button = None
    dropdown = None
    popup = None
    projects = []
    active_project = None
    selected_model = None
    ids = None

    def create_project_name_input_popup(self):
        self.popup = Popup(
            title="New project creation", size_hint=(None, None), size=(400, 150)
        )
        box = BoxLayout(orientation="vertical")

        lbl = Label(text="Please enter new name", size_hint_y=0.3)

        name_input = TextInput(
            text="",
            hint_text="Project name",
            size_hint_y=0.4,
            multiline=False,
            font_size=16,
        )
        name_input.bind(
            on_text_validate=lambda x: self.open_project_folder(
                name_input.text,
            ),
        )

        submit_btn = MDLabelBtn(text="Create", size_hint_y=0.3)
        submit_btn.bind(
            on_release=lambda x: self.open_project_folder(
                name_input.text,
            )
        )
        submit_btn.allow_hover = True

        box.add_widget(lbl)
        box.add_widget(name_input)
        box.add_widget(submit_btn)

        self.popup.content = box
        self.popup.open()

    def setup_project_dropdown(self, projects):
        # If projects folders changed or dropdown was not created
        if projects != self.projects or self.dropdown is None:
            if self.dropdown is not None:
                logger.debug("clear bind")
                self.main_button.unbind(on_release=self.dropdown.open)

            logger.debug("create bind")
            self.dropdown = DropDown()
            for folder in projects:
                btn = Button(text=f"{folder}", size_hint_y=None, height=44)
                btn.bind(on_release=lambda b: self.dropdown.select(b.text))
                self.dropdown.add_widget(btn)

            btn_new = Button(text="New project", size_hint_y=None, height=44)
            btn_new.bind(on_release=lambda b: self.dropdown.select(b.text))
            btn_new.background_color = 0.5, 0.9, 0.5, 1
            self.dropdown.add_widget(btn_new)

            self.main_button.bind(on_release=self.dropdown.open)
            self.dropdown.bind(
                on_select=lambda instance, project: self.open_project_folder(project)
            )
            self.projects = projects

        else:
            logger.debug("use bind")

    def get_projects(self):
        projects = []
        for folder in os.listdir(self.projects_folder):
            if os.path.isdir(os.path.join(self.projects_folder, folder)):
                projects.append(folder)
        return projects

    def open_project_folder(self, project_name):
        logger.debug(f"project_name, {project_name}")
        if project_name == "":
            return
        if self.popup is not None:
            self.popup.dismiss()

        # trigger popup and then call this method again with correct name
        if project_name == "New project":
            self.create_project_name_input_popup()
            return

        self.main_button.text = project_name
        cur_project_path = os.path.join(self.projects_folder, project_name)
        os.makedirs(cur_project_path, exist_ok=True)

        self.after_project_selection_hook(project_name, cur_project_path)

        self.active_project = project_name
        self.restore_project_params(project_name, cur_project_path)

    def after_project_selection_hook(self, project_name, path):
        """
        This is a 'no-op' (no operation) by default.
        Subclasses override this to add custom behavior.
        """
        pass

    def select_model_btn(self, instance):
        logger.info(f"The model button <{instance.text}> is being pressed")
        if self.selected_model:
            if instance.uid == self.selected_model.uid:
                # custom double touch event
                self.unselect_model_btn()
                return

        # reset selection
        for btn in self.ids.model_grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        instance.radius = (20, 20, 20, 20)
        self.selected_model = instance
        self.update_all_button_states()

    def restore_project_params(self, name, path):
        pass

    def update_all_button_states(self):
        pass

    def unselect_model_btn(self):
        pass
