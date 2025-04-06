# --- START OF FILE operator_core.py ---
import os
import json
import asyncio
import traceback
import logging
import os
import sys
import tempfile
import shutil
import re
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Coroutine, Callable
from datetime import datetime
import argparse

# --- Импорт исключений ---
# (v17.1 - без изменений)
try:
    from exceptions import (
        OperatorError, BrowserInteractionError, ElementNotFoundError, NavigationError,
        WaitTimeoutError, FileOperationError, AgentError, AgentResponseError, AgentAPIError
    )
except ImportError:
    class OperatorError(Exception): pass
    class BrowserInteractionError(OperatorError): pass
    class ElementNotFoundError(BrowserInteractionError): pass
    class NavigationError(BrowserInteractionError): pass
    class WaitTimeoutError(BrowserInteractionError): pass
    class FileOperationError(OperatorError): pass
    class AgentError(OperatorError): pass
    class AgentResponseError(AgentError): pass
    class AgentAPIError(AgentError): pass
    _fallback_logger = logging.getLogger("FallbackOperatorLogger")
    _fallback_logger.warning("exceptions.py not found, using fallback exception classes.")

# --- Импорт менеджеров ---
# (v17.1 - без изменений)
try:
    from browser_manager import BrowserManager
except ImportError as e:
    _fallback_logger = logging.getLogger("FallbackOperatorLogger")
    _fallback_logger.critical(f"Критическая ошибка: Не удалось импортировать BrowserManager! {e}", exc_info=True)
    raise ImportError(f"Не удалось импортировать BrowserManager: {e}") from e
try:
    from groq_agent import GroqAgent
except ImportError as e:
    _fallback_logger = logging.getLogger("FallbackOperatorLogger")
    _fallback_logger.critical(f"Критическая ошибка: Не удалось импортировать GroqAgent! {e}", exc_info=True)
    raise ImportError(f"Не удалось импортировать GroqAgent: {e}") from e

# --- Настройка логгера ---
# (v17.1 - без изменений)
logger = logging.getLogger("OperatorAI")
logger.propagate = False
logger.handlers.clear() # Очищаем обработчики по умолчанию, чтобы избежать дублирования

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] - %(message)s')

# Обработчик консоли (как и раньше)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO) # Устанавливаем уровень INFO по умолчанию
logger.addHandler(console_handler)

# Файловый обработчик для журналов уровня DEBUG (данные CBR)
log_file_path = os.path.join(
    tempfile.gettempdir(),
    f"operator_cbr_logs_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
) # Файл журнала во временной директории
file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8') # Режим 'w' для перезаписи при каждом запуске, кодировка 'utf-8'
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG) # Захватываем уровень DEBUG для данных CBR
logger.addHandler(file_handler)

logger.setLevel(logging.INFO) # Устанавливаем уровень INFO по умолчанию

logger.info(f"Путь к файлу журнала CBR: {log_file_path}") # Информируем пользователя о расположении файла журнала

class Operator:
    """
    Основной класс Operator'а с Reflection.
    Версия ~v17.3 (Финальные улучшения обработки ошибок и чистоты кода).
    """
    # <<< ИЗМЕНЕНО: Конструктор __init__ с правильными переносами >>>
    def __init__(self,
                 goal: str,
                 start_url: Optional[str] = None,
                 max_steps: int = 25,
                 step_timeout: int = 180,
                 stuck_detection_threshold: int = 3,
                 summarization_interval: int = 5,
                 reflection_failure_threshold: int = 3,
                 reflection_periodic_interval: int = 10,
                 reflection_history_length: int = 15,
                 dom_max_text_length: int = 1500,
                 dom_max_elements: int = 50,
                 prompt_history_limit_agent: int = 3,
                 browser_headless: bool = True,
                 auto_dismiss_dialogs: bool = True,
                 auto_close_new_tabs: bool = True,
                 agent_type: str = "groq",
                 groq_api_key: Optional[str] = None,
                 groq_model_name: str = "llama3-70b-8192",
                 groq_reflection_model: Optional[str] = None,
                 log_level: str = "INFO"
                 ):
        try:
            
            logger.setLevel(log_level.upper())
        except ValueError:
            logger.warning(f"Неверный log_level '{log_level}'. Используется INFO.")
            logger.setLevel(logging.INFO)

        # <<< ИЗМЕНЕНО: Разделение присваивания атрибутов на отдельные строки >>>
        self.goal: str = goal
        self.start_url: Optional[str] = start_url
        self.max_steps: int = max_steps
        self.step_timeout: int = step_timeout
        self.stuck_detection_threshold: int = stuck_detection_threshold
        self.summarization_interval: int = summarization_interval
        self.reflection_failure_threshold: int = reflection_failure_threshold
        self.reflection_periodic_interval: int = reflection_periodic_interval
        self.reflection_history_length: int = reflection_history_length
        self.dom_max_text_length: int = dom_max_text_length
        self.dom_max_elements: int = dom_max_elements

        self.consecutive_failures: int = 0
        self.steps_since_last_reflection: int = 0
        self.history: List[str] = []
        self.current_step: int = 0
        self.last_retrieved_text: Optional[str] = None
        self.last_extracted_data: Optional[Any] = None
        self.last_step_error: Optional[str] = None
        self.current_plan: Optional[str] = None
        self.scratchpad: str = ""
        self.should_stop: bool = False
        self.workspace_dir: Optional[str] = None
        self.recent_actions_history = deque(maxlen=max(5, stuck_detection_threshold * 2 + 2))
        self.browser_manager: Optional[BrowserManager] = None
        self.ai_agent: Optional[GroqAgent] = None

        try:
            try:
                project_dir = os.path.dirname(os.path.abspath(__file__)) # Получаем папку, где лежит operator_core.py
                self.cbr_db_file = os.path.join(project_dir, "selector_cbr_db.json")
                logger.info(f"Путь к файлу БД CBR: {self.cbr_db_file}")
            except Exception as e:
                logger.error(f"Ошибка при определении пути к файлу БД CBR: {e}")
                self.cbr_db_file = os.path.join(tempfile.gettempdir(), "selector_cbr_db.json")
                logger.info(f"Используется временный путь к файлу БД CBR: {self.cbr_db_file}")
            try:
                logger.debug("Инициализация BrowserManager...")
                # <<< ИЗМЕНЕНО: Перенос аргументов BrowserManager >>>
                self.browser_manager = BrowserManager(
                    browser_type='chromium',
                    headless=browser_headless,
                    default_timeout=25000,
                    retry_attempts=2,
                    auto_dismiss_dialogs=auto_dismiss_dialogs,
                    auto_close_new_tabs=auto_close_new_tabs,
                    cbr_db_filepath=self.cbr_db_file
                )
                logger.debug("BrowserManager инициализирован.")
            except Exception as e:
                logger.critical(f"КРИТ. ОШИБКА при инициализации BrowserManager: {e}", exc_info=True)
                raise OperatorError(f"Ошибка инициализации BrowserManager: {e}") from e

            if agent_type.lower() == "groq":
                 try:
                     logger.info(f"Инициализация GroqAgent (Модель: {groq_model_name})...")
                     effective_groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
                     if not effective_groq_api_key:
                         raise ValueError(f"API ключ Groq не найден (env: GROQ_API_KEY)")
                     # <<< ИЗМЕНЕНО: Перенос аргументов GroqAgent >>>
                     self.ai_agent = GroqAgent(
                         api_key=effective_groq_api_key,
                         model_name=groq_model_name,
                         retry_attempts=3,
                         request_timeout=120,
                         max_tokens_response=2048,
                         reflection_model_name=groq_reflection_model or groq_model_name,
                         prompt_history_limit=prompt_history_limit_agent
                     )
                     logger.debug("GroqAgent инициализирован.")
                 except (ValueError, RuntimeError, AgentError) as e:
                     logger.critical(f"КРИТ. ОШИБКА при инициализации GroqAgent: {e}", exc_info=False)
                     raise OperatorError(f"Ошибка инициализации GroqAgent: {e}") from e
                 except Exception as e:
                     logger.critical(f"КРИТ. НЕОЖИДАННАЯ ОШИБКА при инициализации GroqAgent: {e}", exc_info=True)
                     raise OperatorError(f"Неожиданная ошибка инициализации GroqAgent: {e}") from e
            else:
                raise ValueError(f"Неподдерживаемый тип агента: {agent_type}")

            agent_name = type(self.ai_agent).__name__ if self.ai_agent else 'N/A'
            logger.info(f"Operator инициализирован с {agent_name}.")
            logger.info(f"Параметры: Макс.шагов={self.max_steps}, Таймаут={self.step_timeout}s, Застревание={self.stuck_detection_threshold}")
            logger.info(f"Summarization: Интервал={self.summarization_interval}")
            logger.info(f"Reflection: Порог={self.reflection_failure_threshold}, Интервал={self.reflection_periodic_interval}, История={self.reflection_history_length}")
            agent_history_limit = self.ai_agent.prompt_history_limit if self.ai_agent else 'N/A'
            logger.info(f"DOM: Текст={self.dom_max_text_length}, Элементы={self.dom_max_elements}; История промпта={agent_history_limit}")

        except (OperatorError, FileOperationError, ValueError) as e:
            logger.critical(f"КРИТ. ОШИБКА OperatorError/FileOperationError/ValueError при инициализации: {e}", exc_info=False)
            self._cleanup_sync()
            raise e
        except Exception as e:
            logger.critical(f"КРИТ. НЕПРЕДВИДЕННАЯ ОШИБКА инициализации Operator: {e}", exc_info=True)
            self._cleanup_sync()
            raise RuntimeError(f"Непредвиденная ошибка инициализации Operator: {e}") from e


    # --- МЕТОДЫ _cleanup_sync, _add_history, _check_if_stuck, _handle_password_input, _execute_write_file, _write_file_sync, _request_summary_and_plan_update, _perform_reflection ОСТАЮТСЯ БЕЗ ИЗМЕНЕНИЙ В ЛОГИКЕ ---
    # (Изменены только переносы строк и убраны точки с запятой) ибо я лох и забуду об этом
    def _cleanup_sync(self):
        logger.warning("Синхронная очистка после ошибки инициализации...")
        if self.workspace_dir and os.path.exists(self.workspace_dir):
            try:
                shutil.rmtree(self.workspace_dir)
                logger.info(f"Workspace удален (sync): {self.workspace_dir}")
            except Exception as clean_e:
                logger.error(f"Ошибка синхр. очистки workspace '{self.workspace_dir}': {clean_e}")
        self.workspace_dir = None

    async def _add_history(self, entry_type: str, command: str, args: list, status: str, details: str = ""):
        masked_args = list(args)
        is_password_field_heuristic = False
        if command == "TYPE" and len(masked_args) > 0:
            sel_lower = str(masked_args[0]).lower()
            is_password_field_heuristic = 'password' in sel_lower or 'pwd' in sel_lower or 'pass' in sel_lower

        if command == "TYPE" and len(masked_args) >= 2 :
             text_arg = masked_args[1]
             # <<< ИЗМЕНЕНО: Перенос условия should_mask >>>
             should_mask = (
                 (is_password_field_heuristic and text_arg is not None and str(text_arg).upper() != "PASSWORD") or
                 (text_arg is not None and str(text_arg) == "********")
             )
             if should_mask:
                 masked_args[1] = '********'
             elif text_arg is None:
                 masked_args[1] = '<пропущено>'

        args_repr: str = ""
        try:
            if masked_args:
                args_repr_list = []
                for arg in masked_args:
                    try:
                        s_arg = repr(arg)
                    except Exception:
                        s_arg = "<error repr>"
                    # <<< ИЗМЕНЕНО: Перенос условия в тернарном операторе >>>
                    args_repr_list.append(
                        (s_arg[:97] + '...') if len(s_arg) > 100 else s_arg
                    )
                args_repr = ', '.join(args_repr_list)
        except Exception as e:
            logger.error(f"Неожиданная ошибка форматирования аргументов для {command}: {e}", exc_info=False)
            args_repr = "<error formatting args>"

        details_short = (details[:250] + '...') if len(details) > 253 else details
        details_oneline = details_short.replace('\n', ' ').replace('\r', '')
        log_entry = f"{command}({args_repr}) -> {status.upper()}"
        if details_oneline:
            log_entry += f" [{details_oneline}]"
        logger.info(f"Шаг {self.current_step}: [{entry_type.upper()}] {log_entry}")
        self.history.append(log_entry)
        if entry_type in ["BROWSER", "FILE", "WAIT"]:
            self.recent_actions_history.append((command, args_repr, status.upper()))

    def _check_if_stuck(self) -> Optional[str]:
        if len(self.recent_actions_history) < self.stuck_detection_threshold:
            return None
        last_n_actions = list(self.recent_actions_history)[-self.stuck_detection_threshold:]
        if not all(action[2] == "FAILED" for action in last_n_actions):
            return None
        first_action_cmd_args = (last_n_actions[0][0], last_n_actions[0][1])
        if all((action[0], action[1]) == first_action_cmd_args for action in last_n_actions):
             stuck_cmd, stuck_args_repr = first_action_cmd_args
             # <<< ИЗМЕНЕНО: Перенос длинной f-строки >>>
             warning_msg = (
                 f"ПРЕДУПРЕЖДЕНИЕ_О_ЗАСТРЕВАНИИ: Последние {self.stuck_detection_threshold} "
                 f"действия были одинаковыми ('{stuck_cmd}({stuck_args_repr})') и все FAILED. "
                 f"НЕ ПОВТОРЯЙ! Предложи ДРУГОЙ подход (другой селектор, SCROLL, WAIT_FOR, ASK_USER или FAIL)."
             )
             logger.warning(warning_msg)
             self.recent_actions_history.clear()
             return warning_msg
        return None

    async def _handle_password_input(self, selector: str, text_from_llm: str, original_args: list) -> Tuple[Optional[str], bool]:
        is_password_field = False
        password_marker_used = text_from_llm is not None and text_from_llm.upper() == "PASSWORD"
        text_to_type = text_from_llm
        if self.browser_manager and self.browser_manager.page:
            try:
                locator = await self.browser_manager.find_element(selector, timeout=5000)
                element_type = await locator.evaluate('el => el.type', timeout=1000)
                if element_type == 'password':
                    is_password_field = True
            except (ElementNotFoundError, WaitTimeoutError, BrowserInteractionError) as e:
                logger.debug(f"Не удалось быстро проверить тип поля '{selector}': {e}. Эвристика...")
            except Exception as e:
                logger.warning(f"Ошибка при проверке типа поля '{selector}': {e}")

        if not is_password_field:
            sel_lower = selector.lower()
            is_password_field = 'password' in sel_lower or 'pwd' in sel_lower or 'pass' in sel_lower

        if is_password_field or password_marker_used:
            # <<< ИЗМЕНЕНО: Перенос условия в тернарном операторе >>>
            field_id_suffix = (" (предположительно)" if not is_password_field and password_marker_used else "")
            field_id = f"'{selector}'" + field_id_suffix
            if password_marker_used:
                logger.warning(f"LLM запросил ввод пароля в {field_id} через TYPE. Используйте ASK_USER.")
            else:
                logger.warning(f"Обнаружена попытка ввода текста в поле пароля {field_id}. Автоввод отключен.")
            await self._add_history("OPERATOR", "TYPE", [original_args[0], None], "SKIPPED", f"Автоввод пароля для {field_id} пропущен.")
            return None, True
        else:
            return text_to_type, False

    async def _execute_write_file(self, filename: str, content: str):
        if not self.workspace_dir:
            raise FileOperationError("Workspace не инициализирован.")
        filename = os.path.normpath(filename.strip())
        # <<< ИЗМЕНЕНО: Перенос условия >>>
        if (filename.startswith("..")
                or os.path.isabs(filename)
                or filename.startswith("/")):
            raise ValueError(f"Недопустимый путь к файлу: '{filename}' (должен быть относительным внутри workspace).")

        full_path = os.path.join(self.workspace_dir, filename)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.workspace_dir)):
            raise FileOperationError(f"Попытка записи файла вне workspace: '{filename}'")

        logger.info(f"Запись файла: '{full_path}' ({len(content)} байт)")
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._write_file_sync, full_path, content)
            logger.info(f"Файл '{filename}' успешно записан.")
        except OSError as e:
            logger.error(f"Ошибка OSError при записи '{filename}': {e}")
            raise FileOperationError(f"Ошибка FS при записи '{filename}': {e}") from e
        except Exception as e:
            logger.error(f"Неожиданная ошибка при записи '{filename}': {e}")
            raise FileOperationError(f"Неожиданная ошибка при записи '{filename}': {e}") from e

    def _write_file_sync(self, path: str, content: str):
         try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
         except Exception as e:
             # Re-raise the exception to be caught by the caller
             raise

    async def _request_summary_and_plan_update(self):
        # <<< ИЗМЕНЕНО: Перенос условия >>>
        if (not self.ai_agent
                or not hasattr(self.ai_agent, 'request_summary_and_plan')):
            logger.debug("Агент не инициализирован или не поддерживает summary.")
            return False

        logger.info(f"--- Запрос Суммаризации (Шаг {self.current_step}) ---")
        summary_updated = False
        try:
            # <<< ИЗМЕНЕНО: Перенос условия в тернарном операторе >>>
            history_len_for_summary = (self.summarization_interval * 2
                                       if self.summarization_interval > 0 else 10)
            history_for_summary = self.history[-history_len_for_summary:]

            new_plan, new_scratchpad = await self.ai_agent.request_summary_and_plan(
                self.goal, history_for_summary, self.current_plan, self.scratchpad
            )
            plan_updated = False
            scratchpad_updated = False
            if new_plan is not None and new_plan.strip() and new_plan != self.current_plan:
                logger.info(f"План обновлен (summary):\n{new_plan}")
                self.current_plan = new_plan
                plan_updated = True
            if new_scratchpad is not None and new_scratchpad != self.scratchpad:
                logger.info(f"Блокнот обновлен (summary):\n{new_scratchpad}")
                self.scratchpad = new_scratchpad
                scratchpad_updated = True

            summary_details = [s for s, u in [("План", plan_updated), ("Блокнот", scratchpad_updated)] if u] or ["Изменений нет"]
            await self._add_history("LLM", "SUMMARIZE", [], "EXECUTED", "; ".join(summary_details))
            summary_updated = plan_updated or scratchpad_updated
        except AgentError as e:
            logger.error(f"Ошибка агента при summary: {e}", exc_info=False)
            await self._add_history("OPERATOR", "SUMMARIZE", [], "AGENT_ERROR", str(e))
        except Exception as e:
            logger.error(f"Неожиданная ошибка при summary: {e}", exc_info=True)
            await self._add_history("OPERATOR", "SUMMARIZE", [], "FAILED", str(e))
        return summary_updated

    async def _perform_reflection(self):
        # <<< ИЗМЕНЕНО: Перенос условия >>>
        if (not self.ai_agent
                or not hasattr(self.ai_agent, 'reflect_and_propose_plan')):
            logger.warning("Агент не инициализирован или не поддерживает Reflection.")
            await self._add_history("OPERATOR", "REFLECTION", [], "SKIPPED", "Агент не поддерживает.")
            return

        logger.info(f"--- Запуск Reflection (Шаг {self.current_step}) ---")
        try:
            reflection_history = self.history[-self.reflection_history_length:]
            logger.debug(f"История для Reflection ({len(reflection_history)} шагов).")
            revised_plan = await self.ai_agent.reflect_and_propose_plan(
                self.goal, reflection_history, self.current_plan, self.scratchpad
            )
            if revised_plan and revised_plan.strip():
                 if revised_plan != self.current_plan:
                     logger.info(f"Reflection успешен. Новый план:\n{revised_plan}")
                     self.current_plan = revised_plan
                     await self._add_history("OPERATOR", "REFLECTION", [], "SUCCESS", f"План обновлен ({len(revised_plan)} зн.)")
                 else:
                     logger.info("Reflection: план совпадает с текущим.")
                     await self._add_history("OPERATOR", "REFLECTION", [], "NO_CHANGE", "План не изменился.")
            elif revised_plan is not None: # revised_plan is empty string or whitespace
                logger.warning("Reflection вернул пустой план.")
                await self._add_history("OPERATOR", "REFLECTION", [], "EMPTY_PLAN", "LLM вернул пустой план.")
            else: # revised_plan is None
                logger.warning("Reflection не вернул план (None).")
                await self._add_history("OPERATOR", "REFLECTION", [], "FAILED", "Не удалось получить план (None).")
        except AgentError as e:
            logger.error(f"Ошибка агента при Reflection: {e}", exc_info=False)
            await self._add_history("OPERATOR", "REFLECTION", [], "AGENT_ERROR", str(e))
        except Exception as e:
            logger.error(f"Критическая ошибка при Reflection: {e}", exc_info=True)
            await self._add_history("OPERATOR", "REFLECTION", [], "CRITICAL_FAIL", str(e))

        self.steps_since_last_reflection = 0
        self.consecutive_failures = 0

    # <<< ИЗМЕНЕНО: Переносы строк в _execute_step >>>
    async def _execute_step(self) -> bool:
        """ Выполняет один шаг: Reflection/Summary -> DOM -> LLM -> Действие -> Обработка ошибок (с улучшенным восстановлением). """
        self.current_step += 1
        logger.info(f"--- НАЧАЛО ШАГА {self.current_step}/{self.max_steps} ---")
        step_start_time = asyncio.get_event_loop().time()
        self.steps_since_last_reflection += 1
        stuck_warning_details: Optional[str] = self._check_if_stuck()

        if not self.browser_manager or not self.ai_agent:
            critical_error = OperatorError("КРИТИЧЕСКАЯ ОШИБКА: Менеджеры не были инициализированы!")
            logger.critical(critical_error)
            await self._add_history("OPERATOR", "INTERNAL_ERROR", [], "CRITICAL_FAIL", str(critical_error))
            raise critical_error

        # Reflection / Summarization
        # <<< ИЗМЕНЕНО: Перенос условия should_reflect >>>
        should_reflect = (
            (self.reflection_failure_threshold > 0 and self.consecutive_failures >= self.reflection_failure_threshold) or
            (self.reflection_periodic_interval > 0 and self.steps_since_last_reflection >= self.reflection_periodic_interval)
        )
        if should_reflect:
            await self._perform_reflection()

        # <<< ИЗМЕНЕНО: Перенос условия needs_summary >>>
        needs_summary = (
            not should_reflect and
            self.summarization_interval > 0 and
            (
                ((self.current_step - 1) % self.summarization_interval == 0 and self.current_step > 1) or
                stuck_warning_details # Summarize if stuck warning is present
            )
        )
        if needs_summary:
            await self._request_summary_and_plan_update()

        action_details = ""
        original_args: list = []
        command: Optional[str] = None
        args: list = []
        current_step_error: Optional[Exception] = None
        dom_error_info: Optional[str] = None
        entry_type = "OPERATOR" # Default entry type
        action_successful = False
        action_start_time = asyncio.get_event_loop().time() # Record action start time

        try:
            # 1. Сбор DOM
            logger.debug("Сбор DOM...")
            structured_dom_data = None # Initialize
            try:
                 # <<< ИЗМЕНЕНО: Перенос аргументов get_structured_dom >>>
                 structured_dom_data = await self.browser_manager.get_structured_dom(
                     max_text_length=self.dom_max_text_length,
                     max_elements=self.dom_max_elements
                 )
                 dom_error_info = structured_dom_data.get("error")
                 if dom_error_info:
                     logger.error(f"Ошибка при получении DOM: {dom_error_info}") # Ошибка JS внутри страницы
                 else:
                     logger.debug("DOM успешно собран.")
            except BrowserInteractionError as dom_exc: # Ошибка Python при сборе DOM
                 dom_error_info = f"FAIL_DOM (Python): BrowserInteractionError: {dom_exc}"
                 logger.error(dom_error_info, exc_info=False)
                 structured_dom_data = {"error": dom_error_info}
            except Exception as dom_exc: # Неожиданная ошибка Python
                 dom_error_info = f"FAIL_DOM (Python): Неожиданно: {dom_exc}"
                 logger.critical(dom_error_info, exc_info=True)
                 structured_dom_data = {"error": dom_error_info}
                 # Assign critical error to be raised after history logging
                 current_step_error = RuntimeError(f"Крит. ошибка DOM: {dom_exc}")

            if dom_error_info:
                await self._add_history("BROWSER", "GET_DOM", [], "FAILED", dom_error_info)
                # Set last step error for LLM context
                self.last_step_error = f"DOM Недоступен: {dom_error_info[:200]}"
            if current_step_error:
                # If DOM collection had a critical Python error, raise it now
                raise current_step_error

            # 2. Вызов LLM
            logger.debug("Подготовка контекста и вызов LLM...")
            context_data = structured_dom_data or {"error": "DOM data unavailable"}
            if self.last_step_error:
                context_data['previous_step_error'] = self.last_step_error
                self.last_step_error = None # Clear after passing to LLM
            if stuck_warning_details:
                context_data['stuck_warning'] = stuck_warning_details # Pass stuck warning
            if self.last_retrieved_text is not None:
                context_data['previous_get_text_result'] = self.last_retrieved_text
                self.last_retrieved_text = None # Clear after passing
            if self.last_extracted_data is not None:
                context_data['previous_extracted_data'] = self.last_extracted_data
                self.last_extracted_data = None # Clear after passing

            try:
                 # <<< ИЗМЕНЕНО: Перенос аргументов get_next_action >>>
                 command, args, reasoning, llm_plan_update, llm_scratchpad_update = await self.ai_agent.get_next_action(
                     self.goal,
                     context_data,
                     self.history,
                     self.current_plan,
                     self.scratchpad
                 )
                 original_args = list(args) # Keep original args for history
                 logger.debug(f"LLM предложил команду: {command}")
            except AgentError as agent_exc: # Ошибка API или парсинга ответа LLM
                 logger.error(f"Ошибка AgentError при get_next_action: {agent_exc}", exc_info=False)
                 current_step_error = agent_exc
                 await self._add_history("LLM", "GET_ACTION", [], "AGENT_ERROR", str(agent_exc))
                 raise current_step_error # Propagate error to end step processing

            # Обработка ответа LLM (план, блокнот, рассуждения)
            plan_updated = False
            scratch_updated = False
            if llm_plan_update is not None and llm_plan_update.strip() and llm_plan_update != self.current_plan:
                logger.info(f"План обновлен LLM:\n{llm_plan_update}")
                self.current_plan = llm_plan_update
                plan_updated = True
            if llm_scratchpad_update is not None and llm_scratchpad_update != self.scratchpad:
                logger.info(f"Блокнот обновлен LLM:\n{llm_scratchpad_update}")
                self.scratchpad = llm_scratchpad_update
                scratch_updated = True
            if plan_updated or scratch_updated:
                update_log = [s for s, u in [("План LLM", plan_updated), ("Блокнот LLM", scratch_updated)] if u]
                await self._add_history("LLM", "CONTEXT_UPDATE", [], "EXECUTED", "; ".join(update_log))
            if reasoning:
                logger.info(f"Рассуждение LLM: {reasoning}")
            else:
                logger.warning("LLM не предоставил рассуждение (<reasoning>).")

            # Проверка команды от LLM
            if not command:
                current_step_error = AgentResponseError("Агент не вернул команду.")
                logger.error(current_step_error)
                await self._add_history("LLM", "NO_COMMAND", [], "RESPONSE_ERROR", str(current_step_error))
                raise current_step_error # Propagate error

            if command == "FAIL":
                reason = args[0] if args else "Причина не указана."
                log_reason = reason + (f" || Рассуждение: {reasoning}" if reasoning else "")
                logger.error(f"ЗАДАЧА ПРОВАЛЕНА (LLM FAIL): {log_reason}")
                await self._add_history("LLM", command, original_args, "EXECUTED", log_reason)
                return False # Завершаем выполнение

            # Авто-исправление селектора opid (ТОЛЬКО для команд с селектором!)
            # <<< ИЗМЕНЕНО: Перенос условия >>>
            if command in ["CLICK", "TYPE", "GET_TEXT", "PRESS_KEY", "WAIT_FOR", "EXTRACT_DATA"] and args:
                selector = args[0] # Определяем selector ТОЛЬКО ЗДЕСЬ
                if isinstance(selector, str): # Проверяем, что селектор - строка
                    corrected_selector = None # Инициализируем corrected_selector

                    # Исправление opid-N
                    if re.match(r'^opid-\d+$', selector):
                        corrected_selector = f'[op_id="{selector}"]'
                        logger.debug(f"Автоматически исправлен селектор '{selector}' на '{corrected_selector}'")
                    # Исправление op_id: opid-N
                    elif re.match(r'^op_id:\s*opid-\d+$', selector):
                        opid_value = selector.split(':')[-1].strip()
                        corrected_selector = f'[op_id="{opid_value}"]'
                        logger.debug(f"Автоматически исправлен селектор '{selector}' на '{corrected_selector}'")
                    # <<< ДОБАВЬ ЭТОТ ELIF ДЛЯ :contains >>>
                    elif ':contains(' in selector:
                        # Пытаемся перевести tag:contains('text') или :contains('text') в XPath
                        match = re.match(r"^(.*?):contains\((['\"])(.*?)\2\)$", selector)
                        if match:
                            tag = match.group(1).strip() or '*' # '*' если тег не указан
                            text_content = match.group(3)
                            # Используем normalize-space() для надежности
                            corrected_selector = f"xpath=//{tag}[contains(normalize-space(.), '{text_content}')]"
                            logger.info(f"Переведен селектор '{selector}' в XPath: '{corrected_selector}'")
                        else:
                            logger.warning(f"Не удалось перевести селектор с :contains: '{selector}'. Используется как есть.")
                    # <<< КОНЕЦ БЛОКА ДЛЯ :contains >>>

                    if corrected_selector: # Если селектор был исправлен или переведен
                        args[0] = corrected_selector # Заменяем селектор в аргументах
            if (command in ["CLICK", "TYPE", "GET_TEXT", "PRESS_KEY", "WAIT_FOR", "EXTRACT_DATA"]
                    and args):
                selector = args[0] # Определяем selector ТОЛЬКО ЗДЕСЬ
                if isinstance(selector, str): # Проверяем, что селектор - строка
                    corrected_selector = None # Инициализируем corrected_selector
                    if re.match(r'^opid-\d+$', selector):
                        corrected_selector = f'[op_id="{selector}"]'
                    elif re.match(r'^op_id:\s*opid-\d+$', selector):
                        opid_value = selector.split(':')[-1].strip()
                        corrected_selector = f'[op_id="{opid_value}"]'

                    if corrected_selector: # Если селектор был исправлен
                        logger.debug(f"Автоматически исправлен селектор '{selector}' на '{corrected_selector}'")
                        args[0] = corrected_selector # Заменяем селектор в аргументах
            # --- Конец блока авто-исправления ---

            # 3. Выполнение действия
            logger.debug(f"Попытка выполнить команду: {command} с аргументами: {args}")
            recovery_attempted = False
            is_password_field_flag = False
            # Initialize retry details
            retry_action_func: Optional[Callable[..., Coroutine]] = None
            retry_args_list: List[Any] = []
            retry_kwargs: Dict[str, Any] = {}

            try: # Внутренний try для выполнения команды
                if command == "NAVIGATE":
                    entry_type="BROWSER"
                    await self.browser_manager.navigate(args[0])
                    action_successful=True
                elif command == "CLICK":
                    entry_type = "BROWSER"
                    selector_to_use = None # Инициализируем selector

                    # --- ТЕСТОВЫЙ СПИСОК СЕЛЕКТОРОВ (ЗАКОММЕНТИРОВАНО ПО УМОЛЧАНИЮ) ---
                    # test_selectors = [
                    #     "#non-existent-button", # Нерабочий селектор
                    #     "a[href*='fsf.org']"  # Рабочий селектор (ЗАМЕНИ ЕСЛИ НУЖНО!)
                    # ]
                    # logger.info(f"--- ИСПОЛЬЗУЕТСЯ ТЕСТОВЫЙ СПИСОК СЕЛЕКТОРОВ ДЛЯ CLICK ---")
                    # selector_to_use = test_selectors
                    # --- Конец тестового списка ---

                    # --- Получение селектора от LLM (ЕСЛИ ТЕСТОВЫЙ СПИСОК ЗАКОММЕНТИРОВАН) ---
                    if selector_to_use is None: # Если тестовый список не используется
                        if args:
                            selector_to_use = args[0]
                            logger.info(f"Используем селектор от LLM: {selector_to_use}")
                        else:
                            # Обработка случая, если LLM не дал аргументов для CLICK
                            logger.error("Команда CLICK получена без аргументов! Шаг будет провален.")
                            action_successful = False # Помечаем шаг как неуспешный
                    # --- Конец получения селектора от LLM ---

                    # Вызываем click_element ТОЛЬКО если селектор определен
                    if selector_to_use is not None and action_successful is not False:
                        retry_action_func: Callable = self.browser_manager.click_element # Определяем функцию
                        retry_args_list: List = [selector_to_use] # Аргументы для функции
                        try:
                            await retry_action_func(*retry_args_list) # Вызываем функцию
                            action_successful = True # Успех, если не было исключения
                        except Exception as click_exc:
                            logger.error(f"Ошибка при выполнении CLICK для '{selector_to_use}': {click_exc}")
                            action_successful = False
                    else:
                        logger.error(f"Не удалось определить селектор для команды CLICK.")
                        action_successful = False

                elif command == "TYPE":
                    entry_type = "BROWSER"
                    selector, text_from_llm = args[0], args[1]
                    text_to_type, is_password_field_flag = await self._handle_password_input(selector, text_from_llm, original_args)
                    if text_to_type is not None:
                         retry_action_func = self.browser_manager.type_text
                         retry_args_list = [selector, text_to_type]
                         retry_kwargs = {'log_value': not is_password_field_flag}
                         await retry_action_func(*retry_args_list, **retry_kwargs)
                         action_successful = True
                    else: # Password input skipped
                        action_successful = True # Step considered successful as skipping was intended
                        action_details = "Ввод пароля пропущен."
                elif command == "GET_TEXT":
                    entry_type = "BROWSER"
                    selector = args[0]
                    retry_action_func = self.browser_manager.get_element_text
                    retry_args_list = [selector]
                    text_content = await retry_action_func(*retry_args_list)
                    self.last_retrieved_text = text_content
                    action_successful = True
                    action_details = f"Текст: '{str(text_content)[:100].replace(chr(10),' ')}...'"
                elif command == "SCROLL":
                    entry_type="BROWSER"
                    direction = args[0]
                    pixels = args[1]
                    await self.browser_manager.scroll_page(direction, pixels)
                    action_successful = True
                elif command == "WAIT_FOR":
                    entry_type="WAIT"
                    selector, state = args[0], args[1]
                    timeout_sec = args[2]
                    await self.browser_manager.wait_for_element_state(selector, state, timeout_sec)
                    action_successful = True
                    action_details = f"'{selector}' -> '{state}'."
                elif command == "WRITE_FILE":
                    entry_type="FILE"
                    filename, content = args[0], args[1]
                    await self._execute_write_file(filename, content)
                    action_successful = True
                    action_details = f"Файл '{filename}' записан."
                elif command == "EXTRACT_DATA":
                    entry_type="BROWSER"
                    selector = args[0]
                    data_format = args[1]
                    retry_action_func = self.browser_manager.extract_data_from_element
                    retry_args_list = [selector, data_format]
                    extracted_data = await retry_action_func(*retry_args_list)
                    self.last_extracted_data = extracted_data
                    action_successful = True
                    count = len(extracted_data) if isinstance(extracted_data, list) else 'N/A'
                    action_details = f"Извлечено {count} записей ({data_format})."
                elif command == "PRESS_KEY":
                    entry_type="BROWSER"
                    selector, key_to_press = args[0], args[1]
                    retry_action_func = self.browser_manager.press_key
                    retry_args_list = [selector, key_to_press]
                    await retry_action_func(*retry_args_list)
                    action_successful = True
                    action_details = f"Нажата '{key_to_press}' на '{selector}'."
                elif command == "ASK_USER":
                     entry_type = "USER"
                     question = args[0] if args else "Требуется ввод:"
                     logger.info(f"ЗАПРОС ПОЛЬЗОВАТЕЛЮ: {question}")
                     try:
                         loop = asyncio.get_running_loop()
                         prompt_msg = f"\n--- ВОПРОС ---\n{question}\nОтвет ('stop' для завершения): "
                         user_response = await loop.run_in_executor(None, input, prompt_msg)
                         user_response = user_response.strip()
                         if user_response.lower() == 'stop':
                             logger.info("'stop'. Завершение.")
                             await self._add_history(entry_type, command, original_args, "STOPPED", "Пользователь остановил")
                             self.should_stop = True
                             return False # Stop execution loop
                         elif user_response:
                             self.goal += f"\n[Уточнение]: {user_response}"
                             logger.info(f"Ответ добавлен к цели: {user_response[:100]}...")
                             await self._add_history(entry_type, command, original_args, "RESPONDED", f"Ответ: {user_response[:100]}...")
                             action_successful = True
                             action_details = "Получен ответ."
                         else: # Empty response
                             logger.info("Нет ответа.")
                             await self._add_history(entry_type, command, original_args, "SKIPPED", "Нет ответа")
                             action_successful = True # Still successful, just skipped input
                             action_details = "Запрос пропущен."
                     except Exception as input_e:
                         await self._add_history(entry_type, command, original_args, "FAILED", f"Ошибка ввода: {input_e}")
                         # Re-raise as a specific error type if desired, or just let it propagate
                         raise BrowserInteractionError(f"Ошибка ввода пользователя: {input_e}") from input_e
                elif command == "FINISH":
                    entry_type = "LLM"
                    result = args[0] if args else "Успешно."
                    logger.info(f"ЗАДАЧА ВЫПОЛНЕНА (LLM): {result}")
                    await self._add_history(entry_type, command, original_args, "EXECUTED", result)
                    return False # Stop execution loop
                else:
                    current_step_error = NotImplementedError(f"Команда '{command}' не реализована.")
                    logger.critical(current_step_error)
                    raise current_step_error # This is a critical failure

            # --- Обработка ошибок действия + Восстановление ---
            except (OperatorError, AgentError, RuntimeError) as non_browser_exc:
                # These errors usually mean something is wrong with the operator/agent itself, not the browser state
                logger.error(f"Действие {command} прервано ошибкой оператора/агента: {non_browser_exc}")
                current_step_error = non_browser_exc
                action_details = f"Fail: {str(non_browser_exc)[:150]}."
                # No recovery attempt for these errors
            except (ElementNotFoundError, FileOperationError, ValueError) as specific_fail_exc:
                # Specific failures where reload/retry is unlikely to help
                logger.warning(f"Действие {command} не удалось: {specific_fail_exc}")
                current_step_error = specific_fail_exc
                action_details = f"Fail: {str(specific_fail_exc)[:150]}."
                logger.info(f"Восстановление (Reload+Retry) не применяется для {command} / {type(specific_fail_exc).__name__}.")
            except (NavigationError, WaitTimeoutError, BrowserInteractionError) as browser_exc:
                 # Browser-related errors where reload might help
                 logger.warning(f"Действие {command} не удалось из-за ошибки браузера: {browser_exc}")
                 current_step_error = browser_exc # Keep the original error for context
                 # <<< ИЗМЕНЕНО: Перенос условия can_try_reload_retry >>>
                 can_try_reload_retry = (
                     command in ["CLICK", "TYPE", "GET_TEXT", "PRESS_KEY", "EXTRACT_DATA"] and
                     retry_action_func is not None
                 )
                 if can_try_reload_retry:
                      logger.info(f"Попытка восстановления (Reload+Retry) для {command} после '{type(browser_exc).__name__}'...")
                      recovery_attempted = True
                      action_details = f"Initial fail: {str(browser_exc)[:150]}."
                      try:
                          logger.info("Восстановление: Перезагрузка...")
                          await self.browser_manager.reload_page()
                          logger.info(f"Восстановление: Повтор {command}...")
                          # Повторное выполнение команды
                          if command == "GET_TEXT":
                              text_content = await retry_action_func(*retry_args_list, **retry_kwargs)
                              self.last_retrieved_text = text_content
                              action_details += f" | Reload+Retry OK. Text: '{str(text_content)[:50]}...'"
                          elif command == "EXTRACT_DATA":
                              extracted_data = await retry_action_func(*retry_args_list, **retry_kwargs)
                              self.last_extracted_data = extracted_data
                              count = len(extracted_data) if isinstance(extracted_data, list) else 'N/A'
                              action_details += f" | Reload+Retry OK. Extracted {count} items."
                          else: # CLICK, TYPE, PRESS_KEY
                              await retry_action_func(*retry_args_list, **retry_kwargs)
                              action_details += " | Reload+Retry OK."

                          action_successful = True # Recovery succeeded!
                          current_step_error = None # Clear the error as we recovered
                          logger.info(f"Действие {command} успешно после перезагрузки.")
                      except Exception as recovery_exc:
                           # Error during the recovery attempt
                           logger.warning(f"Восстановление (Reload+Retry) НЕ УДАЛОСЬ: {recovery_exc}")
                           action_details += f" | Reload+Retry FAIL: {str(recovery_exc)[:100]}."
                           # Keep the original browser_exc as current_step_error
                           # Add info about failed recovery to last_step_error for LLM context
                           self.last_step_error = (
                               f"{type(browser_exc).__name__}: {str(browser_exc)[:150]} "
                               f"[Reload+Retry FAILED: {str(recovery_exc)[:100]}]"
                           )
                 else: # Recovery not applicable for this command or error type
                     action_details = f"Fail: {str(browser_exc)[:150]}."
                     logger.info(f"Восстановление (Reload+Retry) не применяется для {command} / {type(browser_exc).__name__}.")

                 # Log final failure if recovery didn't succeed or wasn't attempted
                 if not action_successful:
                     logger.error(f"Действие {command} окончательно НЕ УДАЛОСЬ.")

            # If no error occurred OR recovery was successful, mark as successful
            # Note: action_successful might already be True from the initial try or recovery block
            if not current_step_error:
                action_successful = True

            # If we reach here without returning False (FINISH/FAIL/STOP) or raising an unhandled exception, continue the loop
            return True

        # --- Обработка ошибок, возникших *до* выполнения команды (напр., ошибка LLM, DOM) ---
        except OperatorError as op_error:
            current_step_error = op_error # Capture the error
            logger.error(f"Ошибка OperatorError на шаге {self.current_step}: {op_error}", exc_info=False)
            # Continue the loop, let the error be reported in history/context
            return True
        except Exception as unexpected_exc:
            current_step_error = unexpected_exc # Capture the error
            cmd_name = command if command else 'N/A'
            logger.critical(f"КРИТ. НЕПРЕДВИДЕННАЯ ОШИБКА шага {self.current_step} ({cmd_name}): {unexpected_exc}", exc_info=True)
            # Continue the loop, let the error be reported
            return True

        finally:
            # --- Завершение шага: лог, история, счетчик ошибок ---
            step_duration = asyncio.get_event_loop().time() - step_start_time
            status = "FAILED" # Default status

            if action_successful:
                status = "SUCCESS"
                self.consecutive_failures = 0
            elif current_step_error:
                # Error occurred and wasn't resolved by recovery
                # Set last_step_error if not already set by failed recovery attempt
                if not self.last_step_error:
                    self.last_step_error = f"{type(current_step_error).__name__}: {str(current_step_error)[:250]}"
                self.consecutive_failures += 1
            else:
                # Should not happen if logic is correct (action not successful but no error captured)
                # But handle defensively
                self.last_step_error = f"Шаг {command or '?'} завершился без успеха и без явной ошибки."
                logger.warning(self.last_step_error)
                self.consecutive_failures += 1
                status = "FAILED" # Ensure status is FAILED

            details_for_history = action_details if action_details else ""
            # Append error context to history details if step failed and error context is available
            # <<< ИЗМЕНЕНО: Перенос условия >>>
            if (status == "FAILED"
                    and self.last_step_error
                    and self.last_step_error not in details_for_history): # Avoid duplicate error messages
                 error_msg_for_history = self.last_step_error
                 separator = " | " if details_for_history else ""
                 details_for_history += separator + error_msg_for_history
            elif status == "SUCCESS" and not details_for_history:
                # Provide default success details if none were set
                details_for_history = f"Действие {command or '?'} выполнено."

            # --- Screenshot on Failure ---
            screenshot_path = None # Initialize screenshot path
            # <<< ИЗМЕНЕНО: Перенос условия для скриншота >>>
            if (status == "FAILED" # Only take screenshot on failure
                    and self.browser_manager
                    and self.browser_manager.page
                    and not self.browser_manager.page.is_closed()):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    # Use workspace_dir if available, otherwise temp dir
                    save_dir = self.workspace_dir or tempfile.gettempdir()
                    screenshot_filename = f"step_{self.current_step}_failed_{timestamp}.png"
                    screenshot_path = os.path.join(save_dir, screenshot_filename)
                    await self.browser_manager.take_screenshot(path=screenshot_path, full_page=True, timeout=10000)
                    logger.debug(f"Скриншот ошибки сохранен: '{screenshot_filename}'") # Log relative path if possible
                except Exception as screenshot_e:
                    logger.warning(f"Не удалось сделать скриншот ошибки: {screenshot_e}")
                    screenshot_path = "Не удалось сохранить скриншот" # Indicate failure in log data

            # --- Structured Log for CBR ---
            current_url = None
            if self.browser_manager and self.browser_manager.page and not self.browser_manager.page.is_closed():
                try: current_url = await self.browser_manager.get_current_url()
                except Exception: current_url = "Error getting URL"

            # <<< ИЗМЕНЕНО: Форматирование словаря log_data >>>
            log_data = {
                "step": self.current_step,
                "command": command,
                "args": original_args, # Log original args before modification
                "selector_used": args[0] if args and command in [
                    "CLICK", "TYPE", "GET_TEXT", "PRESS_KEY", "WAIT_FOR", "EXTRACT_DATA"
                    ] else None,
                "status": status,
                "details": details_for_history,
                "error_type": type(current_step_error).__name__ if current_step_error else None,
                "error_message": str(current_step_error) if current_step_error else None,
                "url": current_url,
                "action_duration_sec": asyncio.get_event_loop().time() - action_start_time if action_start_time else None,
                "step_duration_sec": step_duration,
                "screenshot_path": screenshot_path # Add screenshot path to log data
            }
            # Log structured data to DEBUG level file logger
            logger.debug(f"CBR Log Data: {json.dumps(log_data, ensure_ascii=False)}")

            # --- Add to Human-Readable History ---
            # Add history entry unless it was FINISH/STOP (already handled) or command is missing due to error
            # <<< ИЗМЕНЕНО: Перенос условия >>>
            if (command and command not in ["FINISH", "FAIL"] # FINISH/FAIL log their own result
                    and not self.should_stop): # Don't log action if user stopped
                await self._add_history(entry_type, command, original_args, status, details_for_history)
            elif not command and current_step_error: # Log step failure if command wasn't determined
                error_to_log = self.last_step_error or "Неизвестная ошибка шага"
                await self._add_history("OPERATOR", "STEP_FAIL", [], "FAILED", error_to_log)

            # --- Log Step End ---
            logger.info(
                f"--- КОНЕЦ ШАГА {self.current_step} (Статус: {status}, "
                f"Длительность: {step_duration:.2f}s, Ошибок подряд: {self.consecutive_failures}) ---"
            )
            # The return value (True/False) is handled within the try block


    # <<< ИЗМЕНЕНО: Улучшена обработка ошибок и очистка в run + переносы строк >>>
    async def run(self):
        """Основной метод запуска и выполнения оператора."""
        start_time = datetime.now()
        logger.info(f"--- Запуск Оператора ({start_time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
        logger.info(f"Цель: {self.goal}")
        logger.info(f"Старт URL: {self.start_url or 'Нет'}")
        logger.info(f"Параметры: Шаги={self.max_steps}, Таймаут={self.step_timeout}с ...")

        final_status_code = 1 # Код выхода по умолчанию - ПРОВАЛ
        final_message = "Неизвестная ошибка или не завершено"

        try:
            # --- Инициализация браузера и начальная навигация ---
            if not self.browser_manager or not self.ai_agent:
                # Эта ошибка должна была быть поймана в __init__, но проверим еще раз
                raise OperatorError("КРИТИЧЕСКАЯ ОШИБКА: Operator не был корректно инициализирован (менеджеры отсутствуют).")

            browser_start_timeout = max(60.0, self.step_timeout * 0.5)
            logger.info(f"Запуск браузера (таймаут: {browser_start_timeout:.0f}s)...")
            try:
                # Запускаем браузер с таймаутом
                await asyncio.wait_for(
                    self.browser_manager.start_browser(),
                    timeout=browser_start_timeout
                )
                logger.info("Браузер успешно запущен.")
            except asyncio.TimeoutError:
                 logger.critical(f"КРИТ. ОШИБКА ЗАПУСКА: Таймаут ({browser_start_timeout:.0f}s) при запуске браузера!")
                 # Raise a more specific error that indicates browser interaction failure
                 raise BrowserInteractionError(f"Таймаут запуска браузера ({browser_start_timeout:.0f}s)") from None
            except Exception as browser_start_exc: # Ловим другие ошибки start_browser
                 logger.critical(f"КРИТ. ОШИБКА ЗАПУСКА БРАУЗЕРА: {browser_start_exc}", exc_info=True)
                 # Перевыбрасываем как BrowserInteractionError, если это еще не оно, для консистентности
                 if not isinstance(browser_start_exc, BrowserInteractionError):
                      raise BrowserInteractionError(f"Не удалось запустить браузер: {browser_start_exc}") from browser_start_exc
                 else:
                      raise browser_start_exc # Re-raise if it's already the correct type

            # Начальная навигация, если URL указан
            if self.start_url:
                nav_timeout = max(self.step_timeout, 45) # Даем достаточно времени для первой загрузки
                logger.info(f"Переход на старт URL: {self.start_url} (таймаут: {nav_timeout}s)")
                try:
                    await asyncio.wait_for(
                        self.browser_manager.navigate(self.start_url),
                        timeout=nav_timeout
                    )
                    await self._add_history("BROWSER", "NAVIGATE", [self.start_url], "SUCCESS", "На старт URL")
                except asyncio.TimeoutError:
                    logger.error(f"Таймаут ({nav_timeout}s) при переходе на старт URL: {self.start_url}")
                    # Не прерываем полностью, но логируем ошибку. LLM может попробовать снова.
                    await self._add_history("BROWSER", "NAVIGATE", [self.start_url], "FAILED", f"Таймаут навигации ({nav_timeout}s)")
                    self.last_step_error = f"NavigationError: Таймаут ({nav_timeout}s) при переходе на старт URL"
                except NavigationError as nav_exc: # Ловим NavigationError из navigate
                    logger.error(f"Ошибка при переходе на старт URL: {nav_exc}")
                    await self._add_history("BROWSER", "NAVIGATE", [self.start_url], "FAILED", str(nav_exc))
                    self.last_step_error = f"NavigationError: {str(nav_exc)[:200]}"
            else:
                # Log if no start URL is provided
                await self._add_history("OPERATOR", "INIT", [], "INFO", "Старт URL не указан.")

            # --- Основной цикл выполнения шагов ---
            logger.info("--- Начало основного цикла ---")
            while self.current_step < self.max_steps:
                # Add a small buffer to step timeout for overhead
                step_timeout_buffer = 15
                effective_step_timeout = self.step_timeout + step_timeout_buffer
                try:
                    # Выполняем один шаг с общим таймаутом
                    continue_loop = await asyncio.wait_for(
                        self._execute_step(),
                        timeout=effective_step_timeout
                    )
                    if not continue_loop: # Команда FINISH, FAIL или STOP была вызвана
                        last_entry = self.history[-1] if self.history else ""
                        # Определяем финальный статус и сообщение на основе последней записи
                        # <<< ИЗМЕНЕНО: Разделение логики определения final_message >>>
                        if "FINISH" in last_entry and "EXECUTED" in last_entry:
                            final_status_code = 0
                            match = re.search(r"FINISH\(\s*(?:\[\s*['\"]?(.*?)['\"]?\s*\])?\s*\)", last_entry, re.I | re.S)
                            msg = match.group(1).strip() if match and match.group(1) else 'Успешно.'
                            final_message = f"УСПЕХ (FINISH) шаг {self.current_step}: {msg}"
                        elif "STOPPED" in last_entry:
                            final_status_code = 2
                            final_message = f"ОСТАНОВЛЕНО пользователем на шаге {self.current_step}."
                        else: # Assume FAIL if not FINISH or STOPPED
                            final_status_code = 1
                            match = re.search(r"FAIL\(\s*(?:\[\s*['\"]?(.*?)['\"]?\s*\])?\s*\)", last_entry, re.I | re.S)
                            msg = match.group(1).strip() if match and match.group(1) else 'Причина не указана.'
                            final_message = f"ПРОВАЛ (FAIL) шаг {self.current_step}: {msg}"
                        break # Выход из цикла while

                except asyncio.TimeoutError:
                    logger.error(f"КРИТ. ОШИБКА: Таймаут шага {self.current_step} ({effective_step_timeout}s)!", exc_info=False)
                    await self._add_history("OPERATOR", "STEP_TIMEOUT", [], "CRITICAL_FAIL", f"Таймаут шага {effective_step_timeout}s")
                    final_status_code = 1
                    final_message = f"ПРОВАЛ: Таймаут шага {self.current_step}."
                    break # Выход из цикла while
                except OperatorError as oe: # Ошибки, которые _execute_step может пробросить наверх
                    logger.critical(f"КРИТ. ОШИБКА ОПЕРАТОРА (OperatorError) на шаге {self.current_step}: {oe}", exc_info=True)
                    await self._add_history("OPERATOR", "FATAL_ERROR", [], "CRITICAL_FAIL", str(oe))
                    final_status_code = 1
                    final_message = f"КРИТ. OperatorError: {oe}"
                    break # Выход из цикла while
                except Exception as e: # Любые другие неожиданные ошибки на уровне шага
                    logger.critical(f"КРИТ. НЕПРЕДВИДЕННАЯ ОШИБКА ОПЕРАТОРА на шаге {self.current_step}: {e}", exc_info=True)
                    await self._add_history("OPERATOR", "FATAL_ERROR", [], "CRITICAL_FAIL", str(e))
                    final_status_code = 1
                    final_message = f"КРИТ. НЕПРЕДВИДЕННАЯ ОШИБКА: {e}"
                    break # Выход из цикла while

            # Проверка, если вышли из цикла из-за лимита шагов
            # <<< ИЗМЕНЕНО: Перенос условия >>>
            if (self.current_step >= self.max_steps and
                    final_status_code != 0 and # Not already successful
                    final_status_code != 2):   # Not already stopped by user
                logger.warning(f"Достигнут лимит шагов ({self.max_steps}).")
                await self._add_history("OPERATOR", "MAX_STEPS", [], "FAILED", f"Лимит {self.max_steps} шагов.")
                # Update final status only if not already set by FAIL/FINISH/STOP inside loop
                if final_status_code == 1 and final_message == "Неизвестная ошибка или не завершено":
                     final_status_code = 1
                     final_message = f"ПРОВАЛ: Не выполнено за {self.max_steps} шагов."

        # Обработка критических ошибок запуска (до основного цикла)
        except (OperatorError, BrowserInteractionError, NavigationError) as startup_err:
             logger.critical(f"КРИТ. ОШИБКА ЗАПУСКА/НАВИГАЦИИ: {startup_err}", exc_info=True)
             final_status_code = 1
             final_message = f"КРИТ. ОШИБКА ЗАПУСКА: {startup_err}"
        except Exception as general_startup_err:
             logger.critical(f"КРИТ. НЕПРЕДВИДЕННАЯ ОШИБКА ЗАПУСКА: {general_startup_err}", exc_info=True)
             final_status_code = 1
             final_message = f"КРИТ. НЕПРЕДВИДЕННАЯ ОШИБКА ЗАПУСКА: {general_startup_err}"

        # --- Блок Finally для гарантированной очистки ---
        finally:
            end_time = datetime.now()
            total_duration = end_time - start_time
            logger.info(f"--- Завершение ({end_time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
            logger.info(f"Время: {total_duration}")
            logger.info("Закрытие/очистка...")

        # Закрытие браузера
        if hasattr(self, 'browser_manager') and self.browser_manager:
            if hasattr(self.browser_manager, 'save_cbr_db') and hasattr(self, 'cbr_db_file') and self.cbr_db_file:
                self.browser_manager.save_cbr_db(self.cbr_db_file)
            if hasattr(self, 'browser_manager') and self.browser_manager:
                try:
                    await self.browser_manager.close_browser()
                except Exception as close_exc:
                    # Log error during browser close but don't crash
                    logger.error(f"Ошибка при закрытии браузера: {close_exc}", exc_info=True)
            else:
                logger.info("BrowserManager не был инициализирован или уже закрыт.")

            # Удаление workspace
            # <<< ИЗМЕНЕНО: Проверка self.workspace_dir перед os.path.exists >>>
            # <<< ИЗМЕНЕНО: Перенос условия >>>
            if (hasattr(self, 'workspace_dir') and
                    self.workspace_dir and
                    os.path.exists(self.workspace_dir)):
                try:
                    logger.info(f"Очистка workspace: {self.workspace_dir}")
                    shutil.rmtree(self.workspace_dir)
                    logger.info("Workspace удален.")
                except Exception as clean_exc:
                    logger.error(f"Не удалось удалить workspace '{self.workspace_dir}': {clean_exc}")
            elif hasattr(self, 'workspace_dir') and self.workspace_dir:
                # Log if the directory path exists as attribute but not on filesystem
                logger.debug(f"Workspace '{self.workspace_dir}' не найден для удаления.")
            else:
                # Log if workspace was never created or attribute missing
                logger.debug("Workspace не был создан или путь не сохранен.")

            # Итоговое сообщение
            status_text = {
                0: "УСПЕХ",
                1: "ПРОВАЛ",
                2: "ОСТАНОВЛЕНО"
            }.get(final_status_code, "НЕИЗВЕСТНО")
            logger.info("--- ИТОГ ---")
            logger.info(f"Статус: {status_text} (Код: {final_status_code})")
            logger.info(f"Шагов: {self.current_step}")
            logger.info(f"Сообщение: {final_message}")
            logger.info("--- Конец ---")
            return final_status_code # Возвращаем код завершения


# --- Точка входа (__main__) ---
# (Изменены только переносы строк и убраны точки с запятой)
if __name__ == "__main__":
     # Парсер аргументов
     parser = argparse.ArgumentParser(
         description="AI Operator для задач в браузере.",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter
     )
     parser.add_argument("goal", help="Цель оператора.")
     parser.add_argument("-u", "--start-url", default=None, help="Стартовый URL.")
     parser.add_argument("-s", "--max-steps", type=int, default=25, help="Макс. шагов (>= 1).")
     parser.add_argument("-t", "--step-timeout", type=int, default=180, help="Макс. время шага (сек, >= 30).")
     parser.add_argument(
         "-l", "--log-level", default="INFO",
         choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
         help="Уровень лога."
     )
     parser.add_argument(
         "--headless", action=argparse.BooleanOptionalAction, default=True,
         help="Headless режим браузера."
     )
     parser.add_argument(
         "--no-auto-dismiss-dialogs", action="store_false", dest="auto_dismiss_dialogs",
         help="Не отклонять диалоги авто."
     )
     parser.add_argument(
         "--no-auto-close-tabs", action="store_false", dest="auto_close_tabs",
         help="Не закрывать новые вкладки авто."
     )
     parser.add_argument(
         "--groq-api-key", default=None,
         help="Groq API ключ (или env GROQ_API_KEY)."
     )
     parser.add_argument("--groq-model", default="llama3-70b-8192", help="Модель Groq.")
     parser.add_argument(
         "--groq-reflection-model", default=None,
         help="Модель Groq для Reflection (по умолч. = groq-model)."
     )
     parser.add_argument(
         "--summary-interval", type=int, default=5,
         help="Интервал Summarization (шаги, >= 0)."
     )
     parser.add_argument(
         "--stuck-threshold", type=int, default=3,
         help="Порог детекции застревания (>= 2)."
     )
     parser.add_argument(
         "--reflection-threshold", type=int, default=3,
         help="Порог ошибок для Reflection (>= 0)."
     )
     parser.add_argument(
         "--reflection-interval", type=int, default=10,
         help="Интервал Reflection (шаги, >= 0)."
     )
     parser.add_argument(
         "--reflection-history", type=int, default=15,
         help="Длина истории для Reflection (>= 5)."
     )
     parser.add_argument(
         "--dom-max-text", type=int, default=1500,
         help="Макс. длина текста DOM для LLM (>= 500)."
     )
     parser.add_argument(
         "--dom-max-elements", type=int, default=50,
         help="Макс. кол-во элементов DOM для LLM (>= 10)."
     )
     parser.add_argument(
         "--prompt-history-limit", type=int, default=3,
         help="Кол-во последних шагов в истории для промпта LLM (>= 1)."
     )
     args = parser.parse_args()

     # Установка уровня лога
     try:
         logger.setLevel(args.log_level.upper())
     except ValueError:
         logger.setLevel(logging.INFO)
         logger.warning(f"Неверный log_level '{args.log_level}'. Установлен INFO.")

     # Валидация аргументов
     if args.max_steps < 1: parser.error("--max-steps должен быть >= 1")
     if args.step_timeout < 30: parser.error("--step-timeout должен быть >= 30")
     if args.summary_interval < 0: parser.error("--summary-interval должен быть >= 0")
     if args.stuck_threshold < 2: parser.error("--stuck-threshold должен быть >= 2")
     if args.reflection_threshold < 0: parser.error("--reflection-threshold должен быть >= 0")
     if args.reflection_interval < 0: parser.error("--reflection-interval должен быть >= 0")
     if args.reflection_history < 5: parser.error("--reflection-history должен быть >= 5")
     if args.dom_max_text < 500: parser.error("--dom-max-text должен быть >= 500")
     if args.dom_max_elements < 10: parser.error("--dom-max-elements должен быть >= 10")
     if args.prompt_history_limit < 1: parser.error("--prompt-history-limit должен быть >= 1")
     # <<< ИЗМЕНЕНО: Перенос условия проверки ключа API >>>
     if not args.groq_api_key and not os.getenv("GROQ_API_KEY"):
         parser.error("API ключ Groq не найден. Установите переменную окружения 'GROQ_API_KEY' или используйте --groq-api-key.")

     # Запуск Operator
     logger.info("="*40)
     logger.info(f"=== Новый Сеанс Operator AI ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
     logger.info("="*40)
     exit_code = 1 # Default exit code to failure
     operator_instance = None
     try:
          # <<< ИЗМЕНЕНО: Перенос аргументов Operator >>>
          operator_instance = Operator(
              goal=args.goal,
              start_url=args.start_url,
              max_steps=args.max_steps,
              step_timeout=args.step_timeout,
              stuck_detection_threshold=args.stuck_threshold,
              summarization_interval=args.summary_interval,
              reflection_failure_threshold=args.reflection_threshold,
              reflection_periodic_interval=args.reflection_interval,
              reflection_history_length=args.reflection_history,
              dom_max_text_length=args.dom_max_text,
              dom_max_elements=args.dom_max_elements,
              prompt_history_limit_agent=args.prompt_history_limit,
              browser_headless=args.headless,
              auto_dismiss_dialogs=args.auto_dismiss_dialogs,
              auto_close_new_tabs=args.auto_close_tabs,
              agent_type="groq",
              groq_api_key=args.groq_api_key,
              groq_model_name=args.groq_model,
              groq_reflection_model=args.groq_reflection_model,
              log_level=args.log_level
          )
          exit_code = asyncio.run(operator_instance.run())
     except (OperatorError, RuntimeError, ValueError, FileOperationError, BrowserInteractionError) as e:
         
         logger.critical(f"КРИТ. ОШИБКА ЗАПУСКА/ВЫПОЛНЕНИЯ: {e}", exc_info=False)
         exit_code = 1
     except KeyboardInterrupt:
         logger.warning("--- Прервано пользователем (KeyboardInterrupt) ---")
         exit_code = 2 
     except Exception as e:
         
         logger.critical(f"НЕПРЕДВИДЕННАЯ КРИТ. ОШИБКА В __main__: {e}", exc_info=True)
         exit_code = 1
     finally:
        
         logger.info(f"Процесс завершается с кодом {exit_code}")
         logger.info("="*40)
         sys.exit(exit_code)

# --- END OF FILE operator_core.py ---