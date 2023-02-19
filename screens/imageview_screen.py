import os
from shutil import copy

from checksumdir import dirhash
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.filechooser import FileChooserIconView, FileChooserListView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.selectioncontrol import MDCheckbox

from screens.additional import BaseScreen, ImageMDButton, MDLabelBtn
from utils import call_db, extend_key


class ImageViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lock_schedule = False
        self.grid = None
        self.key = ""
        self.selected_images = []
        self.images_to_load = []
        self.progress_bar: ProgressBar = self.ids.progress_bar
        self.load_event = None

        self.loaded_hash = ""
        self.path = "G://Downloads//photo"  # TODO:remove static path

        self.dropdown = None
        self.projects = []

    def on_enter(self, *args):
        self.ids.header.ids[self.manager.current].background_color = 1, 1, 1, 1
        self.key = extend_key(self.manager.get_screen("login").key)
        self.grid = self.ids.grid
        self.selected_counter_update()
        self.create_db_and_check()

        dir_hash = dirhash(self.path, "sha1")

        if self.loaded_hash != dir_hash:
            self.show_folder_images(self.path)

        self.exit_screen = False

    def file_chooser_popup(self):
        def select_directory(file_chooser: FileChooserListView, path):
            path = path if len(path) > 0 else "/"
            if os.path.isdir(path):
                file_chooser.path = path
            else:
                path_label.text = file_chooser.path

        popup = Popup(title="Filechooser", size_hint=(None, None), size=(400, 400))

        box = BoxLayout(orientation="vertical")
        lbl = Label(text="Please select folder", size_hint_y=0.1)

        # chooser = FileChooserListView()
        chooser = FileChooserIconView()

        path_box = BoxLayout(orientation="horizontal", size_hint_y=0.12)

        path_label = TextInput(
            text="", hint_text="path", size_hint_x=0.7, size_hint_y=1, multiline=False
        )
        path_label.bind(
            on_text_validate=lambda x: select_directory(chooser, path_label.text),
        )

        path_btn = MDLabelBtn(text="Submit", size_hint_x=0.3, size_hint_y=1)
        path_btn.allow_hover = True
        path_btn.bind(
            on_press=lambda x: select_directory(chooser, path_label.text),
        )

        path_box.add_widget(path_label)
        path_box.add_widget(path_btn)

        btn = MDLabelBtn(text="Submit", size_hint_y=0.1)
        btn.bind(
            on_press=lambda x: self.show_folder_images(
                chooser.path,
                chooser.selection,
                popup,
            )
        )
        btn.allow_hover = True

        def update_text_field(file_chooser: FileChooserListView, event, p_label):
            if not event.is_mouse_scrolling and not path_label.focus:
                p_label.text = file_chooser.path

        chooser.bind(
            on_touch_up=lambda x, y: Clock.schedule_once(
                lambda tm: update_text_field(x, y, path_label), 0.5
            )
        )

        box.add_widget(lbl)
        box.add_widget(path_box)
        box.add_widget(chooser)
        box.add_widget(btn)

        popup.content = box
        popup.open()

    def show_folder_images(self, path, selection=None, popup=None):
        if popup is not None:
            popup.dismiss()

        self.toggle_load_label("on")

        if os.path.isdir(path):
            files = os.listdir(path)
        else:
            files = None

        self.grid.clear_widgets()
        self.unselect_all_images()
        self.selected_counter_update()

        if files is None:
            self.ids.choose_image.disabled = False
            self.toggle_load_label("no_dir")
            return

        self.ids.choose_image.disabled = True  # disable load button

        for name in files:
            if ".jpg" in name or ".png" in name:
                im_path = os.path.join(path, name)
                self.images_to_load.append(im_path)

        if len(self.images_to_load) == 0:
            self.ids.choose_image.disabled = False
            self.toggle_load_label("no_dir")
            return

        self.progress_bar.value = 1
        self.progress_bar.max = len(self.images_to_load)
        self.load_event = Clock.schedule_interval(
            lambda tm: self.async_image_load(), 0.001
        )

    def async_image_load(self):
        stop = False
        if len(self.images_to_load) == 0:
            self.loaded_hash = dirhash(self.path, "sha1")
            stop = True

        if self.exit_screen:
            print("terminate loading")
            stop = True

        if stop:
            Clock.unschedule(self.load_event)
            self.toggle_load_label("success")
            self.ids.choose_image.disabled = False
            return

        self.progress_bar.value += 1
        im_path = self.images_to_load.pop(0)
        img = ImageMDButton(
            source=im_path,
            allow_stretch=True,
            keep_ratio=True,
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )

        checkbox = MDCheckbox(
            size_hint=(None, None),
            size=("48dp", "48dp"),
            pos_hint={"center_x": 0.96, "center_y": 0.96},
        )

        fl = MDFloatLayout()
        fl.add_widget(img)
        fl.add_widget(checkbox)

        img.line_color = (1.0, 1.0, 1.0, 0.2)
        img.bind(on_press=self.image_click)
        self.grid.add_widget(fl)

    def image_click(self, instance):
        path = instance.source

        if instance in self.selected_images:
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            instance.line_color = (1.0, 1.0, 1.0, 0.2)
            self.selected_images.remove(instance)

            instance.parent.children[0].active = False
        else:
            instance.line_color = (1.0, 1.0, 1.0, 0.6)
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
            self.selected_images.append(instance)

            instance.parent.children[0].active = True

        self.selected_counter_update()

    def save_img_to_db(self, enc):
        from Cryptodome.Cipher import AES

        num_images = len(self.selected_images)
        self.schedule_counter_update()
        if num_images == 0:
            self.ids.selected_images.text = "Choose 1+"
            return

        for path in self.selected_images:
            with open(path.source, "rb") as f:
                blob_data = f.read()

                if enc:
                    cipher = AES.new(self.key, AES.MODE_EAX, nonce=b"TODO")
                    blob_data = cipher.encrypt(blob_data)

                call_db(f"INSERT INTO images (image) VALUES (?)", [blob_data])
        self.unselect_all_images()
        self.ids.selected_images.text = f"Added {num_images}"

    def transfer_images(self, projects_folder, project):
        print(projects_folder, project)

        num_images = len(self.selected_images)

        target_path = os.path.join(projects_folder, project, "all")
        print("target path", target_path)
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
        for path in self.selected_images:
            copy(path.source, target_path)

        self.unselect_all_images()
        self.ids.selected_images.text = f"Copied {num_images}"

    def save_img_to_ml(self):
        num_images = len(self.selected_images)
        self.schedule_counter_update()
        if num_images == 0:
            self.ids.selected_images.text = "Choose 1+"
            return

        app_folder = os.getcwd()
        projects_folder = os.path.join(app_folder, "projects")
        to_ml_btn = self.ids.to_ml_btn

        projects = []
        for folder in os.listdir(projects_folder):
            if os.path.isdir(os.path.join(projects_folder, folder)):
                projects.append(folder)

        # If projects folders changed or dropdown was not created
        if projects != self.projects or self.dropdown is None:
            if self.dropdown is not None:
                print("clear bind")
                to_ml_btn.unbind(on_release=self.dropdown.open)

            print("create bind")
            self.dropdown = DropDown()
            for folder in projects:
                btn = Button(text=folder, size_hint_y=None, height=44)
                btn.bind(on_release=lambda b: self.dropdown.select(b.text))
                self.dropdown.add_widget(btn)

            to_ml_btn.bind(on_release=self.dropdown.open)
            self.dropdown.bind(
                on_select=lambda instance, project: self.transfer_images(
                    projects_folder, project
                )
            )
            self.projects = projects
        else:
            print("use bind")

    def schedule_counter_update(self):
        if not self.lock_schedule:  # to trigger schedule only once at a time
            print("lock")
            self.lock_schedule = True
            Clock.schedule_once(
                lambda dt: self.selected_counter_update(schedule=True), 1
            )

    def select_or_unselect_button_action(self):
        if len(self.selected_images) > 0:
            self.unselect_all_images()
        else:
            self.select_all_images()

    def unselect_all_images(self):
        instances = self.selected_images.copy()
        for instance in instances:
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            instance.line_color = (1.0, 1.0, 1.0, 0.2)
            instance.parent.children[0].active = False
            self.selected_images.remove(instance)
        self.selected_counter_update()

    def select_all_images(self):
        for float_layout in self.grid.children:
            checkbox = float_layout.children[0]
            checkbox.active = True

            image = float_layout.children[1]
            image.line_color = (1.0, 1.0, 1.0, 0.6)
            image.md_bg_color = (1.0, 1.0, 1.0, 0.1)
            self.selected_images.append(image)

        self.selected_counter_update()

    def selected_counter_update(self, schedule=False):
        self.ids.selected_images.text = f"Selected: {len(self.selected_images)}"

        if len(self.selected_images) == 0:
            self.ids.select_unselect_action_button.text = "Select All"
        else:
            self.ids.select_unselect_action_button.text = "Unselect All"

        if schedule:
            self.lock_schedule = False
            print("release")
