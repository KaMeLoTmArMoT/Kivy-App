import os
import webbrowser

from kivy.clock import Clock
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import Screen

from app.screens.services.customer_service import CustomerService
from app.screens.utils.additional import BaseScreen, MDLabelBtn
from app.screens.utils.custom_logging import get_logger
from app.screens.utils.db import DB
from app.screens.utils.utils import extend_key

logger = get_logger(__name__)


class MainScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.customer_service = CustomerService(db=self.db)
        self.key = None
        self.selected = None

        self.delete_btn = self.ids.text_delete
        self.update_btn = self.ids.text_update
        self.url_btn = self.ids.url_open
        self.submit_btn = self.ids.text_submit

        self.chrome_path = None
        self.on_enter_done = False

    def on_enter(self, *args):
        logger.debug("MAIN: on_enter start")
        self.setup_header()
        self.ids.word_input.focus = True
        self.ids.word_input.bind(text=self.on_text_input)
        logger.debug("MAIN: on_enter done")

        Clock.schedule_once(self._finish_enter, 0)

    def _finish_enter(self, dt):
        logger.debug("MAIN: _finish_enter start")
        self.chrome_path = DB().get_config_typed("chrome_path")
        login_key = getattr(self.manager.get_screen("login"), "key", None)
        if not login_key:
            logger.warning("MAIN: _finish_enter skipped due to empty login key")
            return
        self.key = extend_key(login_key)
        self.reload_records()
        self.on_enter_done = True
        logger.debug("MAIN: _finish_enter done")

    def submit(self):
        text = self.get_input()
        self.ids.word_input.text_validate_unfocus = False
        self.unselect_label_btn()

        if len(text) <= 2:
            self.label_out("Text should be longer than 2 letters")
            return

        self.customer_service.add_customer(text, self.key)
        self.reload_records()

        self.label_out(f"{text} added")
        Clock.schedule_once(lambda x: self.label_out("Enter new text:"), 1)

        self.ids.word_input.text = ""

    def reload_records(self):
        records = self.customer_service.get_decrypted_customers(self.key)

        layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        layout.bind(minimum_height=layout.setter("height"))

        for decrypted_name, raw_enc in records:
            btn = MDLabelBtn(text=decrypted_name)
            btn.bind(on_press=self.select_label_btn)
            layout.add_widget(btn)
            self.ids[f"('{raw_enc}',)"] = btn

        self.ids.scroll.clear_widgets()
        self.ids.scroll.add_widget(layout)
        self.unselect_label_btn()
        if len(records) == 0:
            self.label_out("No records. Add any items.")
        else:
            self.label_out("DB instances:")

    def select_label_btn(self, instance):
        logger.info(f"The button <{instance.text}> is being pressed")
        if self.selected and instance.uid == self.selected.uid:
            self.unselect_label_btn()
            return

        if self.ids.scroll.children:
            grid = self.ids.scroll.children[0]
            for btn in grid.children:
                btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        instance.radius = (20, 20, 20, 20)
        self.selected = instance

        self.delete_btn.disabled = False

        if len(self.get_input()) > 2:
            self.update_btn.disabled = False

        if "http" in self.selected.text:
            self.url_btn.disabled = False
        else:
            self.url_btn.disabled = True

    def unselect_label_btn(self):
        self.selected = None

        if self.ids.scroll.children:
            grid = self.ids.scroll.children[0]
            for btn in grid.children:
                btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        self.delete_btn.disabled = True
        self.update_btn.disabled = True
        self.url_btn.disabled = True

    def delete_record(self):
        if self.selected is None:
            self.reload_records()
            self.label_out("First select any element")
            return

        text = self.selected.text
        self.customer_service.delete_customer_by_name(text, self.key)

        self.selected = None
        self.reload_records()
        self.label_out(f"Deleted: {text}")

    def update_record(self):
        if self.selected is None:
            self.reload_records()
            self.label_out("First select any element")
            return

        old_text = self.selected.text
        new_text = self.get_input()

        if len(new_text) <= 2:
            self.label_out("New text should be longer than 2 letters")
            return

        self.customer_service.update_customer_name(old_text, new_text, self.key)

        self.ids.word_input.text = ""
        self.reload_records()
        self.label_out("Successfully updated.")

    def open_url(self):
        if self.selected is None:
            self.label_out("Select link to open")
            return

        url = self.selected.text
        if "http" not in url:
            self.label_out("This is not a link probably")
            return

        if os.name == "posix":
            webbrowser.open(url)
        else:
            webbrowser.get(self.chrome_path + " --incognito").open(url)

    def on_text_input(self, instance, value):
        text = self.get_input()

        self.update_buttons_state()
        if len(text) > 2:
            self.submit_btn.disabled = False
            if self.selected:
                self.update_btn.disabled = False
        else:
            self.submit_btn.disabled = True
            self.update_btn.disabled = True

    def update_buttons_state(self):
        pass
