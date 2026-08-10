from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


def select_model(screen, instance) -> None:
    logger.info(f"The model button <{instance.text}> is being pressed")
    if screen.selected_model and instance.uid == screen.selected_model.uid:
        clear_model(screen)
        return

    for button in screen.ids.model_grid.children:
        button.md_bg_color = (1.0, 1.0, 1.0, 0.0)

    instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
    instance.radius = (20, 20, 20, 20)
    screen.selected_model = instance
    screen.update_all_button_states()


def clear_model(screen) -> None:
    screen.selected_model = None
    if "model_grid" in screen.ids:
        for button in screen.ids.model_grid.children:
            button.md_bg_color = (1.0, 1.0, 1.0, 0.0)
    screen.update_all_button_states()
