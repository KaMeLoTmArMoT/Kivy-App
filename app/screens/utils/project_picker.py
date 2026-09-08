import os
from collections.abc import Callable
from contextlib import suppress

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown, DropDownException
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput

from app.screens.utils.additional import MDLabelBtn


class ProjectPicker:
    """Reusable project dropdown with optional new-project creation."""

    def __init__(
        self,
        button,
        projects_folder: str,
        on_select: Callable[[str], None],
    ):
        self.button = button
        self.projects_folder = projects_folder
        self.on_select = on_select
        self.dropdown = None
        self.popup = None
        self.projects: list[str] = []

    def list_projects(self) -> list[str]:
        if not os.path.isdir(self.projects_folder):
            return []
        return sorted(
            folder
            for folder in os.listdir(self.projects_folder)
            if os.path.isdir(os.path.join(self.projects_folder, folder))
        )

    def open(self) -> None:
        projects = self.list_projects()
        if projects != self.projects or self.dropdown is None:
            self._build_dropdown(projects)
        with suppress(DropDownException):
            self.dropdown.open(self.button)

    def _build_dropdown(self, projects: list[str]) -> None:
        self.dropdown = DropDown()
        for project in [*projects, "New project"]:
            button = Button(text=project, size_hint_y=None, height=44)
            button.bind(on_release=lambda item: self.dropdown.select(item.text))
            if project == "New project":
                button.background_color = 0.5, 0.9, 0.5, 1
            self.dropdown.add_widget(button)

        self.dropdown.bind(on_select=lambda _, project: self._select(project))
        self.projects = projects

    def _select(self, project_name: str) -> None:
        if project_name == "New project":
            self._create_project_popup()
            return
        if project_name:
            self.on_select(project_name)

    def _create_project_popup(self) -> None:
        self.popup = Popup(title="New project creation", size_hint=(None, None), size=(400, 150))
        box = BoxLayout(orientation="vertical")
        name_input = TextInput(
            hint_text="Project name",
            size_hint_y=0.4,
            multiline=False,
            font_size=16,
        )
        submit = MDLabelBtn(text="Create", size_hint_y=0.3, allow_hover=True)
        submit.bind(on_release=lambda *_: self._submit_new_project(name_input))
        name_input.bind(on_text_validate=lambda *_: self._submit_new_project(name_input))

        box.add_widget(Label(text="Please enter new name", size_hint_y=0.3))
        box.add_widget(name_input)
        box.add_widget(submit)
        self.popup.content = box
        self.popup.open()

    def _submit_new_project(self, name_input: TextInput) -> None:
        name = name_input.text.strip()
        if not name:
            return
        if self.popup is not None:
            self.popup.dismiss()
        self.on_select(name)
