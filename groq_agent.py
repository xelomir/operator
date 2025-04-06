# --- START OF FILE groq_agent.py ---
import groq
import os
import re
import asyncio
import logging
import json
import shlex
from dotenv import load_dotenv
from typing import Tuple, List, Optional, Dict, Any, Callable

try:
    from exceptions import WaitTimeoutError, AgentError, AgentResponseError, AgentAPIError
except ImportError:
    # Fallback definitions
    class OperatorError(Exception): pass
    class WaitTimeoutError(Exception): pass
    class AgentError(OperatorError if 'OperatorError' in locals() else Exception): pass
    class AgentResponseError(AgentError): pass
    class AgentAPIError(AgentError): pass
    logger = logging.getLogger("FallbackLogger")
    logger.warning("exceptions.py not found, using fallback exception classes.")

load_dotenv()
logger = logging.getLogger("OperatorAI")

class GroqAgent:
    """
    Класс для взаимодействия с Groq API + Reflection.
    Версия ~v17.9 (Исправлены синтаксические ошибки в _parse_response).
    """
    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: str = "llama3-70b-8192",
                 retry_attempts: int = 3,
                 request_timeout: int = 120,
                 max_tokens_response: int = 2048,
                 reflection_model_name: Optional[str] = None,
                 prompt_history_limit: int = 3
                 ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            env_var_name = "GROQ_API_KEY"
            err_msg = f"API ключ Groq не найден. Установите переменную окружения '{env_var_name}' или передайте аргумент 'api_key' / '--groq-api-key'."
            logger.critical(err_msg)
            raise ValueError(err_msg)
        try:
            key_display = f"...{self.api_key[-4:]}" if len(self.api_key or '') >= 4 else '???'
            logger.debug(f"Попытка инициализации клиента Groq API (ключ: {key_display})")
            self.client = groq.AsyncGroq(api_key=self.api_key)
            logger.info("Клиент Groq API инициализирован.")
        except Exception as e:
            logger.critical(f"Критическая ошибка при инициализации клиента Groq: {e}", exc_info=True)
            raise RuntimeError(f"Ошибка инициализации клиента Groq: {e}") from e

        self.model_name = model_name
        self.retry_attempts = max(1, retry_attempts)
        self.request_timeout = max(10, request_timeout)
        self.max_tokens_response = max(64, max_tokens_response)
        self.reflection_model_name = reflection_model_name or model_name
        self.prompt_history_limit = max(1, prompt_history_limit)
        logger.info(f"GroqAgent: Модель={self.model_name}, Reflection={self.reflection_model_name}, Попытки={self.retry_attempts}, Таймаут={self.request_timeout}s, Макс.Токены={self.max_tokens_response}, Лимит истории={self.prompt_history_limit}")

        self.allowed_commands = ["NAVIGATE", "CLICK", "TYPE", "SCROLL", "GET_TEXT", "WAIT_FOR", "WRITE_FILE", "EXTRACT_DATA", "PRESS_KEY", "FINISH", "ASK_USER", "FAIL"]
        self.expected_args_count: Dict[str, Tuple[int, int]] = {
            "NAVIGATE": (1, 1),
            "CLICK": (1, 1),
            "TYPE": (2, 2),
            "SCROLL": (1, 2),
            "GET_TEXT": (1, 1),
            "WAIT_FOR": (2, 3),
            "WRITE_FILE": (2, 2),
            "EXTRACT_DATA": (1, 2),
            "PRESS_KEY": (2, 2),
            "FINISH": (0, 1),
            "ASK_USER": (0, 1),
            "FAIL": (0, 1)
        }
        self.allowed_wait_states = ['visible', 'hidden', 'enabled', 'disabled', 'editable', 'checked']
        self.allowed_extract_formats = ['list_of_texts', 'table_rows_as_lists', 'table_rows_as_dicts']

    # _format_dom (v17.1 - без изменений)
    def _format_dom(self, structured_dom: Dict[str, Any]) -> str:
        dom_representation = ""
        if not isinstance(structured_dom, dict):
            return "<ОШИБКА: Некорректные данные DOM>"
        if structured_dom.get("error"):
            error_msg = structured_dom['error']
            url_info = structured_dom.get('url', 'N/A')
            dom_representation = f"ОШИБКА СБОРА DOM:\n{error_msg[:500]}...\n URL: {url_info}\n"
            return dom_representation + "===\n"
        url = structured_dom.get('url', 'N/A')
        title = structured_dom.get('title', 'N/A')
        visible_text_raw = structured_dom.get('visible_text','<нет видимого текста>')
        visible_text = ' '.join(visible_text_raw.split())[:1000]
        elements = structured_dom.get('interactive_elements',[])
        dom_representation += f"URL: {url}\nTitle: {title}\n\nВИДИМЫЙ ТЕКСТ (фрагмент):\n{visible_text}...\n\nИНТЕРАКТИВНЫЕ ЭЛЕМЕНТЫ ({len(elements)}):\n"
        if elements:
            for el in elements:
                op_id = el.get('op_id','n/a')
                tag = el.get('tag','?')
                text = el.get('text','').replace('"','\\"').replace('\n',' ')[:80]
                attrs = el.get('attributes',{})
                attr_str = ", ".join(
                    f"{k}='{str(v).replace('\"','\\\"')[:30]}...'" for i, (k, v) in enumerate(attrs.items())
                    if k!='op_id' and v is not None and v != '' and i < 5
                )
                dom_representation += f"- op_id: \"{op_id}\" | <{tag}> | ТЕКСТ: \"{text}...\" | АТРИБ.: {{ {attr_str} }}\n"
        else:
            dom_representation += "<нет интерактивных элементов>\n"
        return dom_representation + "===\n"

    # _construct_prompt_messages (v17.8 - без изменений)
    def _construct_prompt_messages(self, goal: str, context_data: Dict[str, Any], history: list[str], current_plan: Optional[str], current_scratchpad: str) -> list[dict]:
        previous_error = context_data.pop('previous_step_error', None)
        previous_get_text = context_data.pop('previous_get_text_result', None)
        previous_extract_data = context_data.pop('previous_extracted_data', None)
        stuck_warning = context_data.pop('stuck_warning', None)
        system_content = f"""Ты - AI агент "Operator", управляющий браузером для достижения цели.\nТвоя задача - проанализировать ВСЮ информацию, **СТРОГО СЛЕДОВАТЬ ТЕКУЩЕМУ ПЛАНУ ПОШАГОВО**, **продумывая на 1-2 шага вперед**, и выбрать ТОЛЬКО ОДНУ команду для СЛЕДУЮЩЕГО ШАГА.\n\n**ФОРМАТ ОТВЕТА (СТРОГО):**\nИспользуй XML-подобные теги для разделов.\n\n<plan>\n(Опционально: **Полный** обновленный план, если он ИЗМЕНИЛСЯ. Перечисли ВСЕ шаги.)\n1. Шаг 1\n2. Шаг 2\n...\n</plan>\n\n<scratchpad>\n(Опционально: **Полный** обновленный блокнот, если он ИЗМЕНИЛСЯ.)\n...\n</scratchpad>\n\n<reasoning>\n(ОБЯЗАТЕЛЬНО: Рассуждения. **Укажи номер шага плана**, который выполняешь. **Кратко упомяни следующий шаг плана** и логику выбора текущей команды. Обоснуй выбор аргументов. Если отклоняешься от плана из-за ошибки - ОБЪЯСНИ.)\n...\n</reasoning>\n\n(ВАЖНО! СРАЗУ ПОСЛЕ </reasoning>, НА НОВОЙ СТРОКЕ):\n**ОДНА** команда из списка ниже. **БОЛЬШЕ НИЧЕГО** не пиши после команды.\nКОМАНДА(аргумент1, "аргумент2", ...)\n\n**ПРИНЦИПЫ РАБОТЫ:**\n- **СЛЕДУЙ ПЛАНУ, НО ДУМАЙ ВПЕРЕД!** Выполняй текущий шаг, учитывая следующие. Если следующий шаг потребует `SCROLL`, сделай его сейчас.\n- **ПРОАКТИВНО ОБНОВЛЯЙ ПЛАН:** Если ситуация требует изменения плана, обнови раздел `<plan>`.\n- **СКРОЛЛ ПЕРЕД ПОИСКОМ/КЛИКОМ:** Если элемент/информация не видны (смотри DOM), используй `SCROLL("down")` **ПРЕЖДЕ чем** `CLICK`, `TYPE`, `GET_TEXT` или `FAIL`.\n- **РЕАКЦИЯ НА ОШИБКИ:**\n    - Видишь `ОШИБКА ШАГА:<тип>: ...` или `ЗАСТРЕВАНИЕ: ...` -> **НЕ ПОВТОРЯЙ ТО ЖЕ ДЕЙСТВИЕ**.\n    - Ошибка `ElementNotFoundError`: выбери **ДРУГОЙ** селектор (CSS/XPath), попробуй `SCROLL("down")`, или `FAIL`.\n    - Ошибка `TimeoutError`, `NavigationError`, `BrowserInteractionError` (особенно с `Reload+Retry FAIL`): Попробуй `WAIT_FOR`, `SCROLL`, другой селектор или `FAIL`. **НЕ ИСПОЛЬЗУЙ `NAVIGATE`/перезагрузку** без необходимости.\n- **СЕЛЕКТОРЫ:** Предпочитай `[op_id="ид123"]`. Если нет/не сработал -> НАДЕЖНЫЕ CSS (`button[data-testid='login']`, `input#email`) или XPath (`//a[contains(text(), 'Далее')]`). **ПРАВИЛЬНО:** `[op_id="ид123"]`. **НЕПРАВИЛЬНО:** `"ид123"`.\n- **ОВЕРЛЕИ/БАННЕРЫ:** Видишь мешающий элемент -> `CLICK` на кнопке закрытия ('Закрыть', 'Принять', 'X', 'Accept') **ПРЕЖДЕ чем** продолжить. Обнови `<plan>`, если нужно.\n- **АРГУМЕНТЫ КОМАНД:** Строки в **ДВОЙНЫХ** кавычках, разделены запятыми. НЕ `имя=значение`.\n- **ОТПРАВКА ФОРМ:** Используй `PRESS_KEY(selector_поля_ввода, "Enter")` ПОСЛЕ `TYPE`, если возможно.\n\n**ДОСТУПНЫЕ КОМАНДЫ:**\n1.  `NAVIGATE(url: str)`\n2.  `CLICK(selector: str)`\n3.  `TYPE(selector: str, text: str)` (`text="PASSWORD"`)\n4.  `SCROLL(direction: str, [pixels: int])` ('up', 'down', 'top', 'bottom', pixels=500)\n5.  `GET_TEXT(selector: str)`\n6.  `WAIT_FOR(selector: str, state: str, [timeout: int])` ('visible', ..., 'text("...")', timeout=30)\n7.  `WRITE_FILE(filename: str, content: str)`\n8.  `EXTRACT_DATA(selector: str, [format: str])` ('list_of_texts', ..., format='list_of_texts')\n9.  `PRESS_KEY(selector: str, key: str)` ('Enter', ...)\n10. `FINISH([result: str])`\n11. `ASK_USER([question: str])`\n12. `FAIL([reason: str])`\n\n**ПОРЯДОК ВЫВОДА (строго):**\n<plan>...</plan> (Опционально)\n<scratchpad>...</scratchpad> (Опционально)\n<reasoning>...</reasoning> (ОБЯЗАТЕЛЬНО)\nКОМАНДА(...) (ОБЯЗАТЕЛЬНО, НА ПОСЛЕДНЕЙ СТРОКЕ!)\n"""
        user_content_parts = []
        user_content_parts.append(f"ЦЕЛЬ:\n{goal}")
        user_content_parts.append(f"ПЛАН:\n```\n{current_plan or '<План не создан>'}\n```")
        user_content_parts.append(f"БЛОКНОТ:\n```\n{current_scratchpad or '<пусто>'}\n```")
        if previous_error:
            user_content_parts.append(f"**ОШИБКА ПРЕДЫДУЩЕГО ШАГА:**\n{previous_error}")
        if stuck_warning:
            user_content_parts.append(f"**ПРЕДУПРЕЖДЕНИЕ О ЗАСТРЕВАНИИ:**\n{stuck_warning}")
        if previous_get_text:
            user_content_parts.append(f"ПРЕД. GET_TEXT (фрагмент):\n```\n{str(previous_get_text)[:500]}\n```")
        if previous_extract_data:
            try:
                data_str = json.dumps(previous_extract_data, ensure_ascii=False, indent=2)[:800]
            except:
                data_str = str(previous_extract_data)[:800]
            user_content_parts.append(f"ПРЕД. EXTRACT_DATA (фрагмент):\n```\n{data_str}\n```")
        dom_representation = self._format_dom(context_data)
        user_content_parts.append(f"ТЕКУЩИЙ DOM:\n{dom_representation}")
        history_to_show = history[-self.prompt_history_limit:]
        history_section = f"ПОСЛЕДНИЕ ШАГИ ({len(history_to_show)} из {len(history)}):\n```\n{chr(10).join(history_to_show) if history_to_show else '<нет>'}\n```"
        user_content_parts.append(history_section)
        user_content_parts.append("\n**ТВОЙ СЛЕДУЮЩИЙ ШАГ (ВЫВОД СТРОГО ПО ФОРМАТУ):**")
        user_content = "\n---\n".join(user_content_parts)
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
        return messages

    # _construct_summary_prompt_messages, _construct_reflection_messages - (v17.0 - без изменений, используют XML)
    def _construct_summary_prompt_messages(self, goal: str, recent_history: list[str], current_plan: Optional[str], current_scratchpad: str) -> list[dict]:
        system_content = f"""Ты - ИИ-агент "Operator". Твоя задача - **СУММАРИЗИРОВАТЬ** недавнюю историю, текущий план и блокнот, чтобы **обновить и сократить** план и блокнот для дальнейшей работы. Сохраняй только важную информацию и следующие шаги. Будь краток.\n\nИНСТРУКЦИИ:\n1.  Проанализируй историю и текущие план/блокнот.\n2.  Удали выполненные/неактуальные шаги из плана.\n3.  Сократи информацию в блокноте, оставив только релевантное для ЦЕЛИ.\n4.  Сформируй **ПОЛНЫЙ ОБНОВЛЕННЫЙ** план и **ПОЛНЫЙ ОБНОВЛЕННЫЙ** блокнот.\n\n**ПОРЯДОК ВЫВОДА (Оба блока ОБЯЗАТЕЛЬНЫ):**\n<plan>\n1. Шаг 1 (следующий)\n2. Шаг 2\n...\n</plan>\n<scratchpad>\nКраткие заметки...\n</scratchpad>\n\nВАЖНО: Не выводи **НИЧЕГО**, кроме этих двух блоков. План и блокнот должны быть полными.\n"""
        user_content_parts = []
        user_content_parts.append(f"ЦЕЛЬ:\n{goal}")
        user_content_parts.append(f"ТЕКУЩИЙ ПЛАН:\n```\n{current_plan or '<План не создан>'}\n```")
        user_content_parts.append(f"ТЕКУЩИЙ БЛОКНОТ:\n```\n{current_scratchpad or '<пусто>'}\n```")
        user_content_parts.append(f"НЕДАВНЯЯ ИСТОРИЯ ({len(recent_history)} шагов):\n```\n{chr(10).join(recent_history) if recent_history else '<нет>'}\n```")
        user_content_parts.append("\n**ТВОЙ ОБНОВЛЕННЫЙ И СОКРАЩЕННЫЙ ПЛАН И БЛОКНОТ:**")
        user_content = "\n---\n".join(user_content_parts)
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
        return messages

    def _construct_reflection_messages(self, goal: str, reflection_history: list[str], current_plan: Optional[str], current_scratchpad: str) -> list[dict]:
        system_content = f"""Ты - "Рефлексирующий Наблюдатель" для ИИ-агента "Operator". Твоя задача - проанализировать недавний **проблемный** эпизод работы агента и предложить **УЛУЧШЕННЫЙ, ПОЛНЫЙ и КОНКРЕТНЫЙ** план действий для достижения **ЦЕЛИ**.\n\nИНСТРУКЦИИ:\n1.  Проанализируй ЦЕЛЬ, ПЛАН, ИСТОРИЮ (особенно ошибки FAILED и застревания STUCK_WARNING).\n2.  Кратко опиши в `<summary>`: что агент пытался сделать?\n3.  В `<diagnosis>` найди **коренные причины неудач**: Неправильный селектор? Невидимый элемент (нужен SCROLL)? Неверная логика? Игнорирование ошибок?\n4.  В `<plan_critique>` оцени текущий план: актуален ли он? Нужно ли его кардинально менять?\n5.  В `<suggestions>` предложи общие стратегии улучшения (напр., "Использовать более надежные селекторы", "Скроллить перед поиском", "Проверять видимость элемента").\n6.  **ОБЯЗАТЕЛЬНО** в `<revised_plan>` предложи **НОВЫЙ ПОЛНЫЙ И ПОШАГОВЫЙ ПЛАН** от начала до конца (или до `FINISH`/`FAIL`), используя **ТОЛЬКО** разрешенные команды. План должен учитывать ошибки и предлагать КОНКРЕТНЫЕ шаги для обхода проблем.\n\n**РАЗРЕШЕННЫЕ КОМАНДЫ ДЛЯ ПЛАНА:** `NAVIGATE`, `CLICK`, `TYPE`, `SCROLL`, `GET_TEXT`, `WAIT_FOR`, `WRITE_FILE`, `EXTRACT_DATA`, `PRESS_KEY`, `FINISH`, `ASK_USER`, `FAIL`. НЕ ПРИДУМЫВАЙ команды.\n\n**ПОРЯДОК ВЫВОДА (ОБЯЗАТЕЛЬНО):**\n<analysis>\n  <summary>...</summary>\n  <diagnosis>...</diagnosis>\n  <plan_critique>...</plan_critique>\n  <suggestions>...</suggestions>\n</analysis>\n<revised_plan>\n1. КОМАНДА(аргумент1, "аргумент2")\n2. КОМАНДА(...)\n...\n</revised_plan>\n"""
        user_content_parts = []
        user_content_parts.append(f"ЦЕЛЬ АГЕНТА:\n{goal}")
        user_content_parts.append(f"ТЕКУЩИЙ ПЛАН АГЕНТА:\n```\n{current_plan or '<План не создан>'}\n```")
        user_content_parts.append(f"ТЕКУЩИЙ БЛОКНОТ АГЕНТА:\n```\n{current_scratchpad or '<пусто>'}\n```")
        user_content_parts.append(f"НЕДАВНЯЯ ИСТОРИЯ ({len(reflection_history)} шагов):\n```\n{chr(10).join(reflection_history) if reflection_history else '<нет>'}\n```")
        user_content_parts.append("\n**ТВОЙ АНАЛИЗ И ПРЕДЛОЖЕННЫЙ НОВЫЙ ПЛАН:**")
        user_content = "\n---\n".join(user_content_parts)
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
        return messages

    # <<< ИЗМЕНЕНО: Исправлен парсинг и валидация аргументов (if вместо If, добавлено двоеточие) >>>
    def _parse_response(self, response_text: str) -> tuple[Optional[str], List[Any], Optional[str], Optional[str], Optional[str]]:
        response_text = response_text.strip() if response_text else ""
        logger.debug(f"Парсинг ответа LLM (XML формат) (начало):\n{response_text[:500]}...")
        plan_text: Optional[str] = None
        scratchpad_text: Optional[str] = None
        reasoning_text: Optional[str] = None
        command_part: str = response_text
        plan_match = re.search(r"<plan>(.*?)</plan>", response_text, re.DOTALL | re.IGNORECASE)
        scratchpad_match = re.search(r"<scratchpad>(.*?)</scratchpad>", response_text, re.DOTALL | re.IGNORECASE)
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response_text, re.DOTALL | re.IGNORECASE)
        if plan_match:
            plan_text = plan_match.group(1).strip()
            command_part = command_part.replace(plan_match.group(0), '', 1)
            logger.debug("Найден блок <plan>.")
        if scratchpad_match:
            scratchpad_text = scratchpad_match.group(1).strip()
            command_part = command_part.replace(scratchpad_match.group(0), '', 1)
            logger.debug("Найден блок <scratchpad>.")
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            command_part = command_part.replace(reasoning_match.group(0), '', 1)
            logger.debug("Найден блок <reasoning>.")
        else:
            logger.error("! Обязательный блок <reasoning> не найден.")
            return None, [], None, plan_text, scratchpad_text
        command_part = command_part.strip()
        command: Optional[str] = None
        args:List[Any] = []
        raw_args_str = ""
        if not command_part:
            logger.error("! Командная часть ответа LLM отсутствует после извлечения мета-блоков.")
            return None, [], reasoning_text, plan_text, scratchpad_text
        match = re.search(r"\**\s*([a-zA-Z_]+)\s*(?:\((.*)\))?\s*\**\s*$", command_part.strip(), re.DOTALL)
        if match:
            command = match.group(1).upper()
            captured_args = match.group(2)
            raw_args_str = captured_args.strip() if captured_args is not None else ""
            logger.debug(f"Команда '{command}' найдена в конце. Сырые аргументы: '{raw_args_str}'")
        else:
            logger.error(f"! Команда не найдена в конце оставшейся части ответа: '{command_part}'")
            return None, [], reasoning_text, plan_text, scratchpad_text
        if command not in self.allowed_commands:
            logger.error(f"! Недопустимая команда '{command}'. Допустимые: {self.allowed_commands}")
            return None, [], reasoning_text, plan_text, scratchpad_text

        parsed_args: List[Any] = []
        if raw_args_str:
            try:
                lexer = shlex.shlex(raw_args_str, posix=False)
                lexer.whitespace = ','
                lexer.whitespace_split = True
                lexer.quotes = '"\''
                lexer.commenters = ''
                lexer.error_leader = ''
                parsed_tokens = list(lexer)
                temp_args = []
                for token in parsed_tokens:
                    token = token.strip()
                    key_value_match = re.match(r"^\s*\w+\s*=\s*(.*)\s*$", token)
                    value_part = key_value_match.group(1).strip() if key_value_match else token
                    if len(value_part) >= 2 and value_part.startswith(('"', "'")) and value_part.endswith(value_part[0]):
                        temp_args.append(value_part[1:-1])
                    else:
                        try:
                            temp_args.append(int(value_part))
                        except ValueError:
                            try:
                                temp_args.append(float(value_part))
                            except ValueError:
                                temp_args.append(value_part)
                parsed_args = temp_args
                logger.debug(f"Аргументы после shlex: {parsed_args}")
            except ValueError as e:
                logger.warning(f"Ошибка разбора аргументов shlex ('{raw_args_str}'): {e}. Попытка split.")
                parsed_args = [a.strip().strip('"\'') for a in raw_args_str.split(',')]
                logger.debug(f"Аргументы после fallback split: {parsed_args}")
        else:
            parsed_args = []

        min_req, max_req = self.expected_args_count.get(command, (-1, -1))
        if not (min_req <= len(parsed_args) <= max_req):
            logger.error(f"! Неверное кол-во аргументов {len(parsed_args)} для {command} (ожидалось от {min_req} до {max_req}). Арг-ты: {parsed_args}")
            return None, [], reasoning_text, plan_text, scratchpad_text

        validated_args: List[Any] = []
        try:
            # <<< ИСПРАВЛЕНЫ If на if и добавлены двоеточия >>>
            if command == "NAVIGATE":
                validated_args.append(str(parsed_args[0]))
            elif command == "CLICK":
                validated_args.append(str(parsed_args[0]))
            elif command == "TYPE":
                validated_args.append(str(parsed_args[0]))
                validated_args.append(str(parsed_args[1]))
            elif command == "SCROLL":
                direction = str(parsed_args[0]).lower()
                if direction not in ['up', 'down', 'top', 'bottom']:
                    raise ValueError(f"Недопустимое направление SCROLL: {direction}")
                validated_args.append(direction)
                validated_args.append(int(parsed_args[1]) if len(parsed_args) == 2 else 500)
            elif command == "GET_TEXT":
                validated_args.append(str(parsed_args[0]))
            elif command == "WAIT_FOR":
                validated_args.append(str(parsed_args[0]))
                state_arg = str(parsed_args[1]).lower()
                text_match = re.match(r"text\((['\"]?)(.*?)\1\)$", state_arg)
                if text_match:
                    validated_args.append(f'text("{text_match.group(2)}")')
                elif state_arg in self.allowed_wait_states:
                    validated_args.append(state_arg)
                else:
                    raise ValueError(f"Недопустимое состояние WAIT_FOR: {state_arg}")
                validated_args.append(int(parsed_args[2]) if len(parsed_args) == 3 else 30)
            elif command == "WRITE_FILE":
                validated_args.append(str(parsed_args[0]))
                validated_args.append(str(parsed_args[1]))
            elif command == "EXTRACT_DATA":
                validated_args.append(str(parsed_args[0]))
                format_arg = str(parsed_args[1]).lower() if len(parsed_args) == 2 else 'list_of_texts'
                if format_arg in self.allowed_extract_formats:
                    validated_args.append(format_arg)
                else:
                    raise ValueError(f"Недопустимый формат EXTRACT_DATA: {format_arg}")
            elif command == "PRESS_KEY":
                validated_args.append(str(parsed_args[0]))
                validated_args.append(str(parsed_args[1]))
            elif command in ["FINISH", "ASK_USER", "FAIL"]:
                validated_args = [str(arg) for arg in parsed_args]
            else:
                raise NotImplementedError(f"Логика валидации для команды {command} не реализована")
            # <<< КОНЕЦ ИСПРАВЛЕНИЙ >>>
        except (ValueError, TypeError, IndexError, NotImplementedError) as e:
            logger.error(f"! Ошибка валидации/конвертации аргументов для {command}: {e}. Аргументы={parsed_args}")
            return None, [], reasoning_text, plan_text, scratchpad_text

        logger.info(f"Команда успешно распарсена: {command}, Валидированные аргументы: {validated_args}")
        return command, validated_args, reasoning_text, plan_text, scratchpad_text

    # _parse_reflection_response, _parse_summary_response (v17.0 - без изменений)
    def _parse_reflection_response(self, response_text: str) -> Optional[str]:
        response_text = response_text.strip() if response_text else ""
        logger.debug(f"Парсинг ответа Reflection (начало):\n{response_text[:300]}...")
        plan_match = re.search(r"<revised_plan>(.*?)</revised_plan>", response_text, re.DOTALL | re.IGNORECASE)
        if plan_match:
            revised_plan_text = plan_match.group(1).strip()
            logger.info("Найден <revised_plan> в ответе Reflection.")
            return revised_plan_text
        else:
            logger.warning("! Блок <revised_plan> не найден в ответе Reflection!")
            return None

    def _parse_summary_response(self, response_text: str) -> tuple[Optional[str], Optional[str]]:
        response_text = response_text.strip() if response_text else ""
        logger.debug(f"Парсинг ответа Summary (начало):\n{response_text[:300]}...")
        plan_text: Optional[str] = None
        scratchpad_text: Optional[str] = None
        plan_match = re.search(r"<plan>(.*?)</plan>", response_text, re.DOTALL | re.IGNORECASE)
        scratchpad_match = re.search(r"<scratchpad>(.*?)</scratchpad>", response_text, re.DOTALL | re.IGNORECASE)
        if plan_match:
            plan_text = plan_match.group(1).strip()
            logger.debug("Найден <plan> в ответе Summary.")
        else:
            logger.warning("! Блок <plan> не найден в ответе Summary!")
        if scratchpad_match:
            scratchpad_text = scratchpad_match.group(1).strip()
            logger.debug("Найден <scratchpad> в ответе Summary.")
        else:
            logger.warning("! Блок <scratchpad> не найден в ответе Summary!")
        return plan_text, scratchpad_text

    # _make_groq_request (v17.8 - без изменений)
    async def _make_groq_request(self, messages: list[dict], parse_func: Callable[[str], Any], use_reflection_model: bool = False) -> Any:
        last_exception: Optional[Exception] = None
        model_to_use = self.reflection_model_name if use_reflection_model else self.model_name
        parsing_result: Any = None
        for attempt in range(self.retry_attempts):
            logger.info(f"Запрос к Groq ({model_to_use}) (Попытка {attempt + 1}/{self.retry_attempts})...")
            try:
                chat_completion = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        messages=messages,
                        model=model_to_use,
                        temperature=0.1,
                        max_tokens=self.max_tokens_response
                    ),
                    timeout=self.request_timeout
                )
                if not chat_completion.choices:
                    raise groq.APIError(None, body={"message": "Ответ API Groq не содержит 'choices'."}, response=None)
                choice = chat_completion.choices[0]
                response_text = choice.message.content if choice.message else None
                if not response_text:
                    logger.warning(f"Пустой ответ от Groq (choice.message.content is None/empty). Попытка {attempt + 1}.")
                    last_exception = ValueError("Пустой ответ Groq.")
                    await asyncio.sleep(1 + attempt)
                    continue
                finish_reason = choice.finish_reason
                if finish_reason != "stop":
                    logger.warning(f"Groq finish_reason '{finish_reason}' != 'stop'. Возможна неполная генерация.")
                if finish_reason == "length":
                    last_exception = AgentResponseError(f"Ответ Groq обрезан из-за лимита токенов (finish_reason: {finish_reason}).")
                try:
                    parsing_result = parse_func(response_text)
                    logger.debug(f"Результат парсинга (начало): {str(parsing_result)[:300]}...")
                except (ValueError, TypeError, IndexError, AgentResponseError, NotImplementedError, json.JSONDecodeError) as parse_error:
                    logger.error(f"Ошибка парсинга ответа Groq на попытке {attempt + 1}: {parse_error}\nТекст ответа (фрагмент):\n{response_text[:500]}...", exc_info=False)
                    last_exception = AgentResponseError(f"Ошибка парсинга: {parse_error}")
                    await asyncio.sleep(1 + attempt)
                    continue
                if parse_func == self._parse_response:
                    if not isinstance(parsing_result, tuple) or len(parsing_result) != 5:
                        logger.error(f"! _parse_response вернул некорректный тип/длину: {type(parsing_result).__name__}")
                        last_exception = AgentResponseError("Внутренняя ошибка парсинга: некорректный результат _parse_response")
                        await asyncio.sleep(1 + attempt)
                        continue
                    
                    command_parsed = parsing_result[0]
                    reasoning_parsed = parsing_result[2]
                    
                    if command_parsed is None or reasoning_parsed is None:
                        missing_parts = []
                        if command_parsed is None:
                            missing_parts.append('команды')
                        if reasoning_parsed is None:
                            missing_parts.append('тега <reasoning>')
                        missing_str = ' и '.join(missing_parts)
                        logger.warning(f"! Ответ LLM не содержит обязательных частей: {missing_str}.")
                        last_exception = AgentResponseError(f"Ответ LLM не содержит обязательных частей: {missing_str}")
                        await asyncio.sleep(1 + attempt)
                        continue
                
                return parsing_result
            
            except asyncio.TimeoutError:
                logger.warning(f"Таймаут запроса к Groq ({self.request_timeout}s) на попытке {attempt + 1}.")
                last_exception = WaitTimeoutError(f"Таймаут Groq API ({self.request_timeout}s)")
                await asyncio.sleep(1 + attempt)
            
            except groq.RateLimitError as e:
                sleep_time = 10 + attempt * 5
                logger.warning(f"Rate Limit Groq API на попытке {attempt + 1}: {e}. Ожидание {sleep_time} сек...")
                last_exception = AgentAPIError(f"Groq Rate Limit: {e}")
                await asyncio.sleep(sleep_time)
            
            except groq.APIConnectionError as e:
                logger.warning(f"Ошибка соединения Groq API на попытке {attempt + 1}: {e}")
                last_exception = AgentAPIError(f"Groq Connection Error: {e}")
                await asyncio.sleep(3 + attempt)
            
            except groq.InternalServerError as e:
                logger.warning(f"Ошибка сервера Groq (5xx) на попытке {attempt + 1}: {e}")
                last_exception = AgentAPIError(f"Groq Server Error (5xx): {e}")
                await asyncio.sleep(5 + attempt)
            
            except groq.APIStatusError as e:
                logger.error(f"Ошибка статуса Groq API {e.status_code} на попытке {attempt + 1}: {e.message} {e.response.text[:200]}...")
                last_exception = AgentAPIError(f"Groq API Status {e.status_code}: {e.message}")
                if 400 <= e.status_code < 500 and e.status_code != 429:
                    logger.error(f"Неповторяемая ошибка Groq API {e.status_code}. Прерывание попыток.")
                    break
                else:
                    await asyncio.sleep(3 + attempt)
            
            except groq.APIError as e:
                logger.error(f"Общая ошибка Groq API на попытке {attempt + 1}: {e}", exc_info=False)
                last_exception = AgentAPIError(f"Groq API Error: {e}")
                await asyncio.sleep(3 + attempt)
            
            except Exception as e:
                logger.error(f"Неожиданная ошибка при запросе к Groq на попытке {attempt + 1}: {e}", exc_info=True)
                last_exception = e
                break

        logger.error(f"Groq не ответил корректно после {self.retry_attempts} попыток. Последняя ошибка: {last_exception}")
        
        if parse_func == self._parse_response:
            fail_reason = f"LLM/Parser Error ({type(last_exception).__name__}) после {self.retry_attempts} попыток: {str(last_exception)[:150]}"
            reasoning, plan, scratchpad = (parsing_result[2:5] if isinstance(parsing_result, tuple) and len(parsing_result) >= 3 else (None, None, None))
            return "FAIL", [fail_reason], reasoning, plan, scratchpad
        elif parse_func == self._parse_summary_response:
            return (None, None)
        elif parse_func == self._parse_reflection_response:
            return None
        else:
            raise last_exception if last_exception else RuntimeError("Unknown error during Groq request after all retries.")

    # get_next_action, request_summary_and_plan, reflect_and_propose_plan (v17.1 - без изменений)
    async def get_next_action(self, goal: str, context_data: Dict[str, Any], history: list[str], current_plan: Optional[str], current_scratchpad: str) -> tuple[Optional[str], List[Any], Optional[str], Optional[str], Optional[str]]:
        messages = self._construct_prompt_messages(goal, context_data, history, current_plan, current_scratchpad)
        result = await self._make_groq_request(messages, self._parse_response, use_reflection_model=False)
        return result # type: ignore

    async def request_summary_and_plan(self, goal: str, recent_history: list[str], current_plan: Optional[str], current_scratchpad: str) -> tuple[Optional[str], Optional[str]]:
        messages = self._construct_summary_prompt_messages(goal, recent_history, current_plan, current_scratchpad)
        result = await self._make_groq_request(messages, self._parse_summary_response, use_reflection_model=False)
        return result if isinstance(result, tuple) else (None, None)

    async def reflect_and_propose_plan(self, goal: str, reflection_history: list[str], current_plan: Optional[str], current_scratchpad: str) -> Optional[str]:
        messages = self._construct_reflection_messages(goal, reflection_history, current_plan, current_scratchpad)
        revised_plan = await self._make_groq_request(messages, self._parse_reflection_response, use_reflection_model=True)
        return revised_plan if isinstance(revised_plan, str) else None

# --- END OF FILE groq_agent.py ---