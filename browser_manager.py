# --- START OF FILE browser_manager.py ---
import os
import asyncio
import exceptions
import json
import logging
import re
from playwright.async_api import (
    async_playwright, Playwright, Browser, Page, Error, Locator, Dialog,
    TimeoutError as PlaywrightTimeoutError,
    expect
)
from typing import List, Literal, Optional, Dict, Any, Coroutine, Callable, Union 

try:
    from exceptions import (
        OperatorError, BrowserInteractionError, ElementNotFoundError,
        NavigationError, WaitTimeoutError, FileOperationError
    )
except ImportError:
    # Fallback definitions
    class OperatorError(Exception): pass; 
    class BrowserInteractionError(OperatorError): pass; 
    class ElementNotFoundError(BrowserInteractionError): pass; 
    class NavigationError(BrowserInteractionError): pass; 
    class WaitTimeoutError(BrowserInteractionError): pass; 
    class FileOperationError(OperatorError): pass; 
    class AgentError(OperatorError): pass; 
    class AgentResponseError(AgentError): pass; 
    class AgentAPIError(AgentError): pass
    logger = logging.getLogger("FallbackLogger"); logger.warning("exceptions.py not found, using fallback exception classes.")

logger = logging.getLogger("OperatorAI")

class BrowserManager:
    
    """
    Управляет взаимодействием с браузером с помощью Playwright.
    Версия ~v17.7 (Улучшен _handle_new_page для предотвращения закрытия основной страницы и about:blank).
    """
    # __init__ and _retry_async remain the same as v17.6

    def __init__(self,
                 cbr_db_filepath: Optional[str] = None,
                 browser_type: str = 'chromium',
                 headless: bool = True,
                 default_timeout: int = 25000,
                 retry_attempts: int = 2,
                 auto_dismiss_dialogs: bool = True,
                 auto_close_new_tabs: bool = True
                 ):
        self.cbr_db_filepath = cbr_db_filepath
        if self.cbr_db_filepath:
           self.load_cbr_db(self.cbr_db_filepath)
        else:
            logger.warning("Путь к файлу CBR DB не указан, опыт не будет сохранен/загружен.")    
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None # This is our main page
        self.browser_type: str = browser_type
        self.headless: bool = headless
        self.default_timeout: int = default_timeout
        self.auto_dismiss_dialogs: bool = auto_dismiss_dialogs
        self.auto_close_new_tabs: bool = auto_close_new_tabs
        if retry_attempts < 1: logger.warning("retry_attempts < 1. Установлено в 1."); self.retry_attempts: int = 1
        else: self.retry_attempts: int = retry_attempts
        logger.info(f"Инициализация BrowserManager (Браузер: {self.browser_type}, Режим: {'headless' if self.headless else 'GUI'}, Таймаут По Умолч.: {self.default_timeout/1000}s, Попыток Действий: {self.retry_attempts}, AutoDismissDialogs: {self.auto_dismiss_dialogs}, AutoCloseTabs: {self.auto_close_new_tabs})")
        self.selector_performance_db: Dict[str, Dict[str, Any]] = {} # База данных CBR: селектор -> {URL -> {успехи: int, неудачи: int}}
        logger.debug("Инициализирована база данных производительности селекторов (CBR).")
    def load_cbr_db(self, filepath: str):
        """Загружает базу данных CBR из JSON файла."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.selector_performance_db = json.load(f)
                logger.info(f"База данных CBR успешно загружена из '{filepath}'. Записей: {len(self.selector_performance_db)}")
            else:
                logger.info(f"Файл базы данных CBR '{filepath}' не найден. Будет создана новая БД.")
                self.selector_performance_db = {} # Начинаем с пустой, если файла нет
        except (json.JSONDecodeError, IOError, Exception) as e:
            logger.error(f"Ошибка загрузки базы данных CBR из '{filepath}': {e}. Используется пустая БД.", exc_info=False)
            self.selector_performance_db = {} # Используем пустую в случае ошибки
    def save_cbr_db(self, filepath: str):
        """Сохраняет базу данных CBR в JSON файл."""
        if not filepath:
            logger.warning("Не удалось сохранить CBR DB: путь к файлу не указан.")
            return
        if not self.selector_performance_db: # Не сохраняем пустую БД
            logger.info("База данных CBR пуста, сохранение не требуется.")
            return
        try:
            # Создаем директорию, если она не существует
            #dir_path = os.path.dirname(filepath)
            #if dir_path and not os.path.exists(dir_path):
                 #os.makedirs(dir_path, exist_ok=True)
                 #logger.info(f"Создана директория для CBR DB: {dir_path}")

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.selector_performance_db, f, ensure_ascii=False, indent=2)
            logger.info(f"База данных CBR успешно сохранена в '{filepath}'. Записей: {len(self.selector_performance_db)}")
        except (IOError, TypeError, Exception) as e:
            logger.error(f"Ошибка сохранения базы данных CBR в '{filepath}': {e}", exc_info=False)        

    
    # _retry_async (v17.4 - без изменений)
    async def _retry_async(self, async_func: Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs) -> Any:
        last_exception: Optional[Exception] = None
        retry_logic_timeout_ms = kwargs.pop('timeout', self.default_timeout)
        total_timeout_sec = (retry_logic_timeout_ms / 1000.0) + 2.0
        internal_kwargs = kwargs
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"Вызов '{async_func.__name__}' (Попытка {attempt + 1}/{self.retry_attempts}) с общим таймаутом {total_timeout_sec:.1f}s...")
                result = await asyncio.wait_for(async_func(*args, **internal_kwargs), timeout=total_timeout_sec)
                logger.debug(f"'{async_func.__name__}' успешно завершен на попытке {attempt + 1}.")
                return result
            except asyncio.TimeoutError:
                last_exception = WaitTimeoutError(f"Общий таймаут {total_timeout_sec:.1f}s для '{async_func.__name__}' истек на попытке {attempt + 1}.")
                logger.warning(last_exception)
                # Continue retries for consistency with other exceptions
            except PlaywrightTimeoutError as e: logger.warning(f"Попытка {attempt + 1}/{self.retry_attempts}: Внутренний таймаут Playwright у '{async_func.__name__}'. {e}"); last_exception = e
            except (ElementNotFoundError, NavigationError, WaitTimeoutError, BrowserInteractionError) as e: logger.warning(f"Попытка {attempt + 1}/{self.retry_attempts}: Ошибка браузера у '{async_func.__name__}'. {type(e).__name__}: {e}"); last_exception = e
            except Error as e: logger.warning(f"Попытка {attempt + 1}/{self.retry_attempts}: Ошибка Playwright у '{async_func.__name__}'. {type(e).__name__}: {e}"); last_exception = BrowserInteractionError(f"Ошибка Playwright: {e}")
            except Exception as e: logger.error(f"Попытка {attempt + 1}/{self.retry_attempts}: Неожиданная ошибка у '{async_func.__name__}'. {type(e).__name__}: {e}", exc_info=True); last_exception = e; break
            if attempt < self.retry_attempts - 1: wait_time = 1.0 + attempt; logger.debug(f"Ожидание {wait_time:.1f}s перед следующей попыткой..."); await asyncio.sleep(wait_time)
        logger.error(f"Операция '{async_func.__name__}' не удалась после {self.retry_attempts} попыток.")
        final_exception: Exception
        if last_exception is None: final_exception = BrowserInteractionError(f"Операция '{async_func.__name__}' завершилась без успеха и без явного исключения после {self.retry_attempts} попыток.")
        elif isinstance(last_exception, (WaitTimeoutError, BrowserInteractionError, ElementNotFoundError, NavigationError)): final_exception = last_exception
        elif isinstance(last_exception, PlaywrightTimeoutError): final_exception = WaitTimeoutError(f"Таймаут Playwright у '{async_func.__name__}' после {self.retry_attempts} попыток: {last_exception}")
        else: final_exception = BrowserInteractionError(f"Неожиданная ошибка у '{async_func.__name__}' после {self.retry_attempts} попыток: {last_exception}")
        raise final_exception

    # _handle_dialog (v17.1 - без изменений)
    async def _handle_dialog(self, dialog: Dialog):
        logger.warning(f"Обнаружено диалоговое окно: тип={dialog.type}, сообщение='{dialog.message()}'. Авто-отклонение.")
        try: await dialog.dismiss(); logger.info("Диалоговое окно отклонено.")
        except Error as e: logger.error(f"Ошибка при отклонении диалогового окна: {e}")

    # <<< ИЗМЕНЕНО: Улучшен _handle_new_page >>>
    async def _handle_new_page(self, new_page: Page):
        """Обработчик для автоматического закрытия новых страниц (вкладок/окон)."""
        # Небольшая пауза, чтобы страница успела инициализироваться и получить URL
        await asyncio.sleep(0.5)

        # Проверяем, не является ли эта страница основной страницей оператора
        if new_page is self.page:
            logger.debug("Обработчик _handle_new_page вызван для основной страницы - игнорируем закрытие.")
            return

        page_url = "N/A"
        try:
              # Пропускаем 'about:blank', если это не единственная страница
             page_url = new_page.url or "N/A"

             # Игнорируем 'about:blank', если только это не единственная открытая страница (маловероятно)
             if page_url == 'about:blank' and len(new_page.context.pages) > 1:
                  logger.debug(f"Обнаружена новая страница 'about:blank'. Игнорируем авто-закрытие.")
                  # Можно попытаться ее закрыть, но это может вызвать гонки. Пока просто игнорируем.
                  # try:
                  #      await new_page.close()
                  #      logger.debug("Страница 'about:blank' закрыта.")
                  # except Error as close_blank_err:
                  #      logger.warning(f"Не удалось закрыть 'about:blank': {close_blank_err}")
                  return

             logger.warning(f"Обнаружена новая страница/вкладка: URL='{page_url}'. Авто-закрытие.")
             await new_page.close()
             logger.info(f"Новая страница ({page_url}) закрыта.")
        except Error as e: # Ошибка Playwright при получении URL или закрытии
            # Игнорируем ошибку 'Target page, context or browser has been closed', т.к. это ожидаемо, если что-то закрыло ее раньше
            if "Target page, context or browser has been closed" in str(e):
                 logger.debug(f"Не удалось закрыть новую страницу ({page_url}), так как она уже была закрыта: {e}")
            else:
                 logger.error(f"Ошибка Playwright при авто-закрытии новой страницы ({page_url}): {e}")
        except Exception as e: # Другие ошибки
            logger.error(f"Неожиданная ошибка при обработке новой страницы ({page_url}): {e}", exc_info=True)


    # start_browser (v17.6 - без изменений)
    async def start_browser(self) -> Page:
        if self.page and not self.page.is_closed(): logger.warning("Браузер уже запущен."); return self.page
        logger.info("Запуск Playwright и браузера...")
        try:
            async def _launch_internal():
                try:
                    if not self.playwright: logger.debug("Запуск экземпляра Playwright..."); self.playwright = await async_playwright().start(); logger.debug("Экземпляр Playwright запущен.")
                    browser_launcher = getattr(self.playwright, self.browser_type)
                    launch_options = {'headless': self.headless}
                    logger.debug(f"Запуск браузера {self.browser_type} с опциями: {launch_options}...")
                    self.browser = await browser_launcher.launch(**launch_options)
                    pid_info = ""
                    try:
                        if self.browser and self.browser.process: pid_info = f"(PID: {self.browser.process.pid})"
                    except AttributeError: logger.warning("Атрибут 'process' или 'pid' не найден у объекта Browser.")
                    except Exception as pid_exc: logger.warning(f"Не удалось получить PID браузера: {pid_exc}")
                    logger.debug(f"Браузер {self.browser_type} запущен {pid_info}.")
                    context = await self.browser.new_context(bypass_csp=True); logger.debug("Контекст браузера создан.")
                    if self.auto_close_new_tabs: context.on("page", lambda page: asyncio.create_task(self._handle_new_page(page))); logger.info("Активирован обработчик авто-закрытия новых вкладок.")
                    self.page = await context.new_page(); logger.debug("Новая страница создана.") # <<< self.page присваивается здесь
                    self.page.set_default_timeout(self.default_timeout)
                    if self.auto_dismiss_dialogs: self.page.on("dialog", lambda dialog: asyncio.create_task(self._handle_dialog(dialog))); logger.info("Активирован обработчик авто-отклонения диалогов.")
                    logger.info(f"Браузер '{self.browser_type}' и страница успешно запущены. Таймаут по умолч.: {self.default_timeout / 1000} сек.")
                    return self.page
                except Error as playwright_err:
                    if "Executable doesn't exist" in str(playwright_err): logger.critical(f"КРИТ. ОШИБКА PLAYWRIGHT: Не найден исполняемый файл браузера {self.browser_type}! Возможно, требуется 'playwright install {self.browser_type}'. Ошибка: {playwright_err}", exc_info=False); raise BrowserInteractionError(f"Исполняемый файл браузера не найден. Запустите 'playwright install {self.browser_type}'.") from playwright_err
                    else: logger.critical(f"Крит. ошибка Playwright при запуске браузера: {playwright_err}", exc_info=True); raise BrowserInteractionError(f"Ошибка Playwright при запуске: {playwright_err}") from playwright_err
                except Exception as e: logger.critical(f"Крит. НЕОЖИДАННАЯ ошибка при запуске браузера: {e}", exc_info=True); raise BrowserInteractionError(f"Неожиданная ошибка запуска: {e}") from e

            launch_timeout = max(60000, self.default_timeout * 2)
            return await self._retry_async(_launch_internal, timeout=launch_timeout)

        except BrowserInteractionError as bie: logger.critical(f"КРИТ. ОШИБКА ЗАПУСКА БРАУЗЕРА: {bie}", exc_info=False); await self.close_browser(); raise RuntimeError(f"Критическая ошибка запуска браузера: {bie}") from bie
        except Exception as e: logger.critical(f"Крит. ошибка механизма запуска браузера (_retry_async): {e}", exc_info=True); await self.close_browser(); raise RuntimeError(f"Критическая ошибка механизма запуска браузера: {e}") from e

    # --- Остальные методы без изменений (v17.3) ---
    async def navigate(self, url: str):
        if not self.page: raise RuntimeError("Страница не инициализирована.")
        if not url.startswith(('http://', 'https://')): url = f"https://{url}"
        logger.info(f"Навигация по URL: {url}")
        nav_timeout = max(self.default_timeout * 2, 45000)
        try:
            async def _goto():
                 # <<< Увеличим немного внутренний таймаут goto на всякий случай >>>
                 internal_goto_timeout = nav_timeout + 5000
                 response = await self.page.goto(url, wait_until='domcontentloaded', timeout=internal_goto_timeout)
                 if response and not response.ok: logger.warning(f"Страница {url} загружена со статусом {response.status}.")
                 elif not response: logger.warning(f"Навигация по {url} вернула None ответ (возможно, редирект или ошибка).")
                 return response
            # <<< Увеличим общий таймаут retry для навигации >>>
            await self._retry_async(_goto, timeout=nav_timeout + 10000)
            current_url = self.page.url
            logger.info(f"Успешно перешел на: {current_url} (исходный URL: {url})")
        except Exception as e:
            logger.error(f"Ошибка навигации по {url} после всех попыток: {e}")
            if isinstance(e, NavigationError): raise e
            # <<< Добавим обертку для TargetClosedError >>>
            elif "Target page, context or browser has been closed" in str(e):
                 raise NavigationError(f"Ошибка навигации по {url}: Страница была неожиданно закрыта.") from e
            else: raise NavigationError(f"Ошибка навигации по {url}: {e}") from e

    async def get_current_url(self) -> Optional[str]:
        if not self.page or self.page.is_closed():
            logger.warning("Не удалось получить URL: страница закрыта или не инициализирована.")
            return None
        try:
            return self.page.url
        except Error as e:
            logger.warning(f"Не удалось получить URL (ошибка Playwright): {e}")
            return None

    async def get_content(self) -> Optional[str]:
        if not self.page or self.page.is_closed():
            logger.error("Не удалось получить контент: страница закрыта.")
            return None # <<< Добавлено is_closed() check
        logger.info("Получение HTML контента страницы...")
        try:
            content = await self._retry_async(self.page.content, timeout=self.default_timeout)
            logger.info(f"HTML контент получен (длина: {len(content)}).")
            return content
        except Exception as e:
            logger.error(f"Не удалось получить контент страницы: {e}")
            return None

    async def find_element(self, selector: Union[str, List[str]], timeout: Optional[int] = None) -> Locator: # <--- ИЗМЕНЕН ТИП selector
        if not self.page or self.page.is_closed(): raise ElementNotFoundError(f"Страница закрыта, поиск невозможен.")
        effective_timeout = timeout or self.default_timeout

        # --- Преобразование selector в список для цикла ---
        selectors_to_try: List[str] = []
        if isinstance(selector, str):
            selectors_to_try = [selector]
        elif isinstance(selector, list):
            selectors_to_try = selector
        else:
            raise TypeError(f"Неверный тип для 'selector': ожидался str или List[str], получен {type(selector).__name__}")

        logger.info(f"Поиск элемента (селекторы: {selectors_to_try}) (таймаут видимости: {effective_timeout/1000}s)")

        # --- Получение URL для CBR ---
        current_url = "unknown_url" # Значение по умолчанию
        try:
            current_url = self.page.url # Получаем текущий URL
        except Error as url_error:
            logger.warning(f"Не удалось получить текущий URL для CBR: {url_error}")

        last_exception: Optional[Exception] = None  # Для сохранения последней ошибки
        successful_selector_used: Optional[str] = None  # Для отслеживания, какой селектор сработал

        # --- ПРИОРИТЕЗАЦИЯ СЕЛЕКТОРОВ с помощью CBR (ШАГ 3.3) ---
        if len(selectors_to_try) > 1: # Сортируем, только если селекторов больше одного
            def get_selector_score(sel):
                score = 0.0
                if sel in self.selector_performance_db and current_url in self.selector_performance_db[sel]:
                    stats = self.selector_performance_db[sel][current_url]
                    successes = stats.get("successes", 0)
                    failures = stats.get("failures", 0)
                    # Простая формула: +1 за успех, -1 за неудачу, 0 если нет данных
                    score = successes - failures
                else:
                    score = 0 # Нейтральный скор для селекторов без истории на этом URL
                logger.debug(f"CBR Score для '{sel}' на '{current_url}': {score}")
                return score

            try:
                selectors_to_try.sort(key=get_selector_score, reverse=True) # Сортируем по убыванию скора (лучшие сначала)
                logger.info(f"Селекторы отсортированы CBR для URL '{current_url}': {selectors_to_try}")
            except Exception as sort_err:
                logger.error(f"Ошибка сортировки селекторов CBR: {sort_err}", exc_info=False)
        # --- КОНЕЦ ПРИОРИТЕЗАЦИИ СЕЛЕКТОРОВ ---

        # === НАЧАЛО ЦИКЛА FOR ===
        for current_selector in selectors_to_try:
            logger.debug(f"Попытка селектора: '{current_selector}'")
            locator = None  # Сбрасываем locator перед каждой попыткой

            # --- Основной блок try для поиска и fallback ---
            try:
                # 1. ПЕРВОНАЧАЛЬНАЯ ПОПЫТКА: Поиск по CSS (или текущему селектору)
                try:
                    locator = self.page.locator(current_selector).first
                    internal_wait_timeout = effective_timeout + 5000 # Увеличиваем внутренний таймаут
                    async def _wait_visible(): await locator.wait_for(state='visible', timeout=internal_wait_timeout) # Используем увеличенный таймаут
                    await self._retry_async(_wait_visible, timeout=effective_timeout + 1000)
                    logger.info(f"Элемент найден и видим (Селектор: '{current_selector}').")
                    successful_selector_used = current_selector

                    # --- ОБНОВЛЕНИЕ CBR (УСПЕХ) (ШАГ 3.2) ---
                    try:
                        if current_selector not in self.selector_performance_db:
                            self.selector_performance_db[current_selector] = {}
                        if current_url not in self.selector_performance_db[current_selector]:
                            self.selector_performance_db[current_selector][current_url] = {"successes": 0, "failures": 0}
                        self.selector_performance_db[current_selector][current_url]["successes"] += 1
                        logger.debug(f"CBR DB обновлена: УСПЕХ для '{current_selector}' на '{current_url}'")
                    except Exception as db_update_err:
                        logger.error(f"Ошибка обновления CBR DB (успех): {db_update_err}", exc_info=False)
                    # --- КОНЕЦ ОБНОВЛЕНИЯ CBR (УСПЕХ) ---

                    return locator # Успех! Возвращаем элемент.

                except ElementNotFoundError as initial_error:
                    logger.warning(f"Селектор '{current_selector}' не найден изначально: {initial_error}. Попытка Scroll+Retry.")
                    last_exception = initial_error # Сохраняем первую ошибку

                    # 2. ВТОРАЯ ПОПЫТКА: "Прокрутка и Повтор"
                    try:
                        await self.scroll_page(direction='down', pixels=300) # Прокрутка вниз
                        await asyncio.sleep(0.5) # Пауза после прокрутки
                        locator = self.page.locator(current_selector).first # Повторный поиск ПОСЛЕ прокрутки
                        internal_wait_timeout_scroll = effective_timeout + 5000 # Увеличиваем внутренний таймаут
                        async def _wait_visible_after_scroll(): await locator.wait_for(state='visible', timeout=internal_wait_timeout_scroll) # Используем увеличенный таймаут
                        await self._retry_async(_wait_visible_after_scroll, timeout=effective_timeout + 1000)
                        logger.info(f"Элемент найден после Scroll+Retry (Селектор: '{current_selector}').")
                        successful_selector_used = current_selector

                        # --- ОБНОВЛЕНИЕ CBR (УСПЕХ ПОСЛЕ SCROLL) (ШАГ 3.2) ---
                        try:
                            if current_selector not in self.selector_performance_db:
                                self.selector_performance_db[current_selector] = {}
                            if current_url not in self.selector_performance_db[current_selector]:
                                self.selector_performance_db[current_selector][current_url] = {"successes": 0, "failures": 0}
                            self.selector_performance_db[current_selector][current_url]["successes"] += 1
                            logger.debug(f"CBR DB обновлена: УСПЕХ (Scroll) для '{current_selector}' на '{current_url}'")
                        except Exception as db_update_err:
                            logger.error(f"Ошибка обновления CBR DB (успех scroll): {db_update_err}", exc_info=False)
                        # --- КОНЕЦ ОБНОВЛЕНИЯ CBR (УСПЕХ ПОСЛЕ SCROLL) ---

                        return locator # Успех после Scroll+Retry! Возвращаем элемент.

                    except ElementNotFoundError as scroll_retry_error:
                        logger.warning(f"Scroll+Retry не помог для '{current_selector}': {scroll_retry_error}. Попытка XPath fallback.")
                        last_exception = scroll_retry_error # Сохраняем ошибку Scroll+Retry

                        # 3. ТРЕТЬЯ ПОПЫТКА: XPath Fallback
                        try:
                            # --- Улучшенный XPath ---
                            # Используем current_selector как текст для поиска в XPath
                            xpath_selector = f"""xpath=//*[normalize-space(text())='{current_selector}' or @aria-label='{current_selector}' or @value='{current_selector}']"""
                            logger.debug(f"Попытка XPath fallback селектора: '{xpath_selector}'")
                            locator = self.page.locator(xpath_selector).first
                            internal_wait_timeout_xpath = effective_timeout + 5000 # Увеличиваем внутренний таймаут
                            async def _wait_visible_xpath(): await locator.wait_for(state='visible', timeout=internal_wait_timeout_xpath) # Используем увеличенный таймаут
                            await self._retry_async(_wait_visible_xpath, timeout=effective_timeout + 1000)
                            logger.info(f"Элемент найден и видим (XPath селектор '{xpath_selector}' fallback после Scroll+Retry).")
                            successful_selector_used = xpath_selector # Помечаем, что XPath сработал

                            # --- ОБНОВЛЕНИЕ CBR (УСПЕХ ПОСЛЕ XPATH) (ШАГ 3.2) ---
                            try:
                                # Обновляем статистику для ИСХОДНОГО селектора, но можно и для XPath
                                if current_selector not in self.selector_performance_db:
                                    self.selector_performance_db[current_selector] = {}
                                if current_url not in self.selector_performance_db[current_selector]:
                                    self.selector_performance_db[current_selector][current_url] = {"successes": 0, "failures": 0}
                                self.selector_performance_db[current_selector][current_url]["successes"] += 1 # Считаем успех для исходного
                                logger.debug(f"CBR DB обновлена: УСПЕХ (XPath) для '{current_selector}' на '{current_url}'")
                                # Можно отдельно считать статистику для XPath селекторов, если нужно
                                # if xpath_selector not in self.selector_performance_db: self.selector_performance_db[xpath_selector] = {}
                                # ... и т.д.
                            except Exception as db_update_err:
                                logger.error(f"Ошибка обновления CBR DB (успех xpath): {db_update_err}", exc_info=False)
                            # --- КОНЕЦ ОБНОВЛЕНИЯ CBR (УСПЕХ ПОСЛЕ XPATH) ---

                            return locator # Успех после XPath fallback! Возвращаем элемент.

                        except ElementNotFoundError as xpath_fallback_error:
                            logger.error(f"XPath fallback '{xpath_selector}' тоже не сработал после Scroll+Retry: {xpath_fallback_error}")
                            last_exception = xpath_fallback_error # Сохраняем ошибку XPath fallback

                            # --- ОБНОВЛЕНИЕ CBR (НЕУДАЧА ПОСЛЕ ВСЕХ ПОПЫТОК) (ШАГ 3.2) ---
                            try:
                                if current_selector not in self.selector_performance_db:
                                    self.selector_performance_db[current_selector] = {}
                                if current_url not in self.selector_performance_db[current_selector]:
                                    self.selector_performance_db[current_selector][current_url] = {"successes": 0, "failures": 0}
                                self.selector_performance_db[current_selector][current_url]["failures"] += 1
                                logger.debug(f"CBR DB обновлена: НЕУДАЧА (CSS+Scroll+XPath) для '{current_selector}' на '{current_url}'")
                            except Exception as db_update_err:
                                logger.error(f"Ошибка обновления CBR DB (неудача xpath): {db_update_err}", exc_info=False)
                            # --- КОНЕЦ ОБНОВЛЕНИЯ CBR (НЕУДАЧА ПОСЛЕ ВСЕХ ПОПЫТОК) ---

                            continue # Переходим к следующему селектору в списке

                        except Exception as xpath_fallback_exception: # Ловим ДРУГИЕ ошибки XPath fallback
                            logger.error(f"Ошибка при XPath fallback после Scroll+Retry: {xpath_fallback_exception}", exc_info=True)
                            last_exception = xpath_fallback_exception
                            continue # Переходим к следующему селектору в списке

                    except Exception as retry_exception: # Ловим ДРУГИЕ ошибки Scroll+Retry
                        logger.error(f"Ошибка при Scroll+Retry для '{current_selector}': {retry_exception}", exc_info=True)
                        last_exception = retry_exception
                        continue # Переходим к следующему селектору в списке

                except Exception as e: # Ловим ЛЮБЫЕ ДРУГИЕ ошибки ПЕРВОНАЧАЛЬНОЙ попытки (кроме ElementNotFoundError)
                    logger.error(f"Неожиданная ошибка при использовании селектора '{current_selector}': {e}", exc_info=True)
                    last_exception = e # Сохраняем ошибку
                    continue # Переходим к следующему селектору в списке

            except Exception as outer_error: # Ловим ВНЕШНИЕ ошибки (если вдруг что-то пойдет совсем не так)
                 logger.critical(f"Критическая внешняя ошибка в цикле find_element для '{current_selector}': {outer_error}", exc_info=True)
                 last_exception = outer_error
                 continue # Переходим к следующему селектору

        # === КОНЕЦ ЦИКЛА FOR ===

        # --- Если ни один селектор из списка не сработал ---
        selector_list_str = str(selectors_to_try) # Отображаем весь список опробованных селекторов
        error_message = f"Не удалось найти элемент по селекторам {selector_list_str} после всех попыток (CSS, Scroll+Retry, XPath)."
        logger.error(error_message)

        if last_exception: # Если была сохранена какая-либо ошибка
            if isinstance(last_exception, ElementNotFoundError):
                # Добавляем контекст к существующей ошибке ElementNotFoundError
                raise ElementNotFoundError(f"{error_message} Последняя ошибка: {last_exception}") from last_exception
            else:
                # Оборачиваем другую ошибку в ElementNotFoundError
                raise ElementNotFoundError(f"{error_message} Последняя ошибка: {type(last_exception).__name__}: {last_exception}") from last_exception
        else:
            # Если по какой-то причине last_exception не установлена
            raise ElementNotFoundError(error_message + " Не удалось определить конкретную ошибку.")

            

    async def click_element(self, selector: str, timeout: Optional[int] = None):
        if not self.page or self.page.is_closed(): raise BrowserInteractionError(f"Не удалось кликнуть '{selector}': страница закрыта.") # <<< Добавлено is_closed() check
        effective_timeout = timeout or self.default_timeout
        click_internal_timeout = int(effective_timeout * 0.8)
        async def _click_internal():
            element = await self.find_element(selector, timeout=effective_timeout)
            logger.info(f"Клик по элементу: '{selector}'")
            await element.click(timeout=click_internal_timeout)
            logger.info(f"Клик по '{selector}' выполнен.")
        try: await self._retry_async(_click_internal, timeout=effective_timeout)
        except Exception as e:
            logger.error(f"Ошибка при клике на '{selector}': {e}")
            if isinstance(e, (ElementNotFoundError, BrowserInteractionError)): raise e
            else: raise BrowserInteractionError(f"Клик на '{selector}' не удался: {e}") from e

    async def type_text(self, selector: str, text: str, delay: int = 50, timeout: Optional[int] = None, clear_before: bool = True, log_value: bool = True):
        if not self.page or self.page.is_closed(): raise BrowserInteractionError(f"Не удалось ввести текст в '{selector}': страница закрыта.") # <<< Добавлено is_closed() check
        effective_timeout = timeout or self.default_timeout
        type_internal_timeout = int(effective_timeout * 0.8); expect_timeout = max(1000, int(effective_timeout / 3))
        log_text_display = text if log_value else '********'; log_text_short = (log_text_display[:50] + '...') if len(log_text_display) > 53 else log_text_display
        async def _type_internal():
            element = await self.find_element(selector, timeout=effective_timeout)
            try: await expect(element).to_be_editable(timeout=expect_timeout)
            except PlaywrightTimeoutError: raise BrowserInteractionError(f"Элемент '{selector}' не редактируемый за {expect_timeout}ms.")
            logger.info(f"Ввод текста '{log_text_short}' в: '{selector}'")
            if clear_before: await element.fill("", timeout=type_internal_timeout)
            await element.fill(text, timeout=type_internal_timeout)
            logger.info(f"Текст '{log_text_short}' введен в '{selector}'.")
        try: await self._retry_async(_type_internal, timeout=effective_timeout)
        except Exception as e:
            logger.error(f"Ошибка при вводе текста в '{selector}': {e}")
            if isinstance(e, (ElementNotFoundError, BrowserInteractionError)): raise e
            else: raise BrowserInteractionError(f"Ввод текста в '{selector}' не удался: {e}") from e

    async def get_element_text(self, selector: str, timeout: Optional[int] = None) -> str:
        if not self.page or self.page.is_closed(): raise BrowserInteractionError(f"Не удалось получить текст из '{selector}': страница закрыта.") # <<< Добавлено is_closed() check
        effective_timeout = timeout or self.default_timeout
        get_text_internal_timeout = int(effective_timeout * 0.8)
        async def _get_text_internal():
             element = await self.find_element(selector, timeout=effective_timeout)
             logger.info(f"Получение текста из элемента: '{selector}'")
             text = await element.text_content(timeout=get_text_internal_timeout)
             result = text.strip() if text else ""
             display_text = result.replace('\n', ' ')[:60] + ('...' if len(result) > 60 else '')
             logger.info(f"Текст из '{selector}': '{display_text}'")
             return result
        try: return await self._retry_async(_get_text_internal, timeout=effective_timeout)
        except Exception as e:
            logger.error(f"Ошибка при получении текста из '{selector}': {e}")
            if isinstance(e, (ElementNotFoundError, BrowserInteractionError)): raise e
            else: raise BrowserInteractionError(f"Получение текста из '{selector}' не удалось: {e}") from e

    async def scroll_page(self, direction: Literal['up', 'down', 'bottom', 'top'], pixels: int = 500):
        if not self.page or self.page.is_closed(): logger.warning("Не удалось скроллить: страница закрыта."); return # <<< Добавлено is_closed() check
        if direction not in ['up', 'down', 'bottom', 'top']: logger.warning(f"Неизвестное направление скролла '{direction}'. Используйте 'up', 'down', 'bottom', 'top'."); return
        log_details = f"{direction}" + (f", {pixels}px" if direction in ['up', 'down'] else ""); logger.info(f"Прокрутка страницы: {log_details}")
        js_code = ""
        if direction == 'down': js_code = f'window.scrollBy(0, {pixels})'
        elif direction == 'up': js_code = f'window.scrollBy(0, -{pixels})'
        elif direction == 'bottom': js_code = 'window.scrollTo(0, document.body.scrollHeight)'
        elif direction == 'top': js_code = 'window.scrollTo(0, 0)'
        try: await self.page.evaluate(js_code); logger.info(f"Скролл ({log_details}) выполнен.")
        except Error as e: logger.error(f"Ошибка при выполнении JS для скролла ({log_details}): {e}")
        except Exception as e: logger.error(f"Неожиданная ошибка при скролле ({log_details}): {e}", exc_info=True)

    async def reload_page(self):
        if not self.page or self.page.is_closed(): raise NavigationError("Не удалось перезагрузить: страница закрыта.") # <<< Добавлено is_closed() check
        logger.info("Перезагрузка страницы..."); reload_timeout = max(self.default_timeout * 2, 45000)
        try:
            async def _reload(): await self.page.reload(wait_until='domcontentloaded', timeout=reload_timeout)
            await self._retry_async(_reload, timeout=reload_timeout + 5000); logger.info("Страница успешно перезагружена.")
        except Exception as e:
             logger.error(f"Ошибка при перезагрузке страницы: {e}")
             if isinstance(e, NavigationError): raise e
             else: raise NavigationError(f"Перезагрузка не удалась: {e}") from e

    async def get_structured_dom(self, max_text_length=1500, max_elements=50) -> Dict[str, Any]:
        if not self.page or self.page.is_closed(): logger.error("Крит. ошибка get_structured_dom: Страница закрыта."); return {"error": "CRITICAL_PYTHON_ERROR: Страница закрыта."}
        logger.info(f"Получение структурированного DOM (max_text={max_text_length}, max_el={max_elements})..."); id_prefix = "opid"
        script = f"""(() => {{ const MAX_TEXT_LENGTH = {max_text_length}; const MAX_ELEMENTS = {max_elements}; const ID_PREFIX = "{id_prefix}"; const result = {{ url: '', title: '', visible_text: '', interactive_elements: [], error: null }}; try {{ result.url = window.location.href; result.title = document.title; if (!document.body) {{ result.error = "JS_WARNING: document.body не найден."; return JSON.stringify(result); }} const textNodes = []; const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false); let node; while(node = walk.nextNode()) {{ const parentElement = node.parentElement; if (parentElement) {{ const style = window.getComputedStyle(parentElement); if (style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0') {{ const tagName = parentElement.tagName.toUpperCase(); if (tagName !== 'SCRIPT' && tagName !== 'STYLE' && tagName !== 'NOSCRIPT' && !parentElement.hasAttribute('aria-hidden') && parentElement.offsetParent !== null) {{ const text = node.nodeValue.trim(); if (text) textNodes.push(text); }} }} }} }} result.visible_text = textNodes.join(' ').replace(/\\s+/g, ' ').substring(0, MAX_TEXT_LENGTH); const interactiveTags = ['a', 'button', 'input:not([type="hidden"])', 'select', 'textarea', 'label', '[role="button"]', '[role="link"]', '[role="checkbox"]', '[role="radio"]', '[role="tab"]', '[role="menuitem"]', '[tabindex]:not([tabindex="-1"])']; const elements = Array.from(document.body.querySelectorAll(interactiveTags.join(','))); const visibleElementsMap = new Map(); let elementCount = 0; let currentId = 0; for (const el of elements) {{ if (elementCount >= MAX_ELEMENTS) break; try {{ const style = window.getComputedStyle(el); const rect = el.getBoundingClientRect(); if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0' || rect.width === 0 || rect.height === 0 || el.closest('[aria-hidden="true"]') || el.offsetWidth === 0 || el.offsetHeight === 0) {{ continue; }} const uniqueId = `${{ID_PREFIX}}-${{currentId++}}`; el.setAttribute('op_id', uniqueId); const elementData = {{ op_id: uniqueId, tag: el.tagName.toLowerCase(), text: (el.textContent || el.value || el.placeholder || el.getAttribute('aria-label') || '').trim().replace(/\\s+/g, ' ').substring(0, 100), attributes: {{}} }}; const attrs_to_collect = ['id', 'class', 'name', 'type', 'role', 'aria-label', 'placeholder', 'value', 'href', 'title', 'alt', 'for', 'aria-selected', 'aria-checked', 'aria-expanded', 'aria-disabled', 'data-testid']; for (const attr of attrs_to_collect) {{ const val = el.getAttribute(attr); if (val !== null && val !== '') {{ elementData.attributes[attr] = val.substring(0, 50); }} }} if (el.disabled !== undefined) {{ elementData.attributes['disabled'] = el.disabled; }} if (el.checked !== undefined) {{ elementData.attributes['checked'] = el.checked; }} if (el.readOnly !== undefined) {{ elementData.attributes['readonly'] = el.readOnly; }} visibleElementsMap.set(uniqueId, elementData); elementCount++; }} catch (elementError) {{ console.warn(`Operator DOM Script: Ошибка элемента: ${{elementError.message}}`, el); }} }} result.interactive_elements = Array.from(visibleElementsMap.values()); }} catch (e) {{ result.error = `JS_ERROR: ${{e.name}} - ${{e.message}}`; console.error("Operator DOM Script Error:", e); }} return JSON.stringify(result);}})();"""
        try:
            dom_script_timeout = int(self.default_timeout * 1.5)
            json_string_result = await self._retry_async(self.page.evaluate, script, timeout=dom_script_timeout)
            if not isinstance(json_string_result, str): error_msg = f"PYTHON_ERROR: Скрипт DOM вернул не строку ({type(json_string_result).__name__})"; logger.error(error_msg); return {"error": error_msg}
            structured_data = json.loads(json_string_result)
            if structured_data.get("error"): logger.error(f"Ошибка JS при получении DOM: {structured_data['error']}"); return structured_data
            else: logger.info(f"Структурированный DOM получен ({len(structured_data.get('interactive_elements', []))} интерактивных эл-тов)."); return structured_data
        except json.JSONDecodeError as e: error_fragment = str(json_string_result)[:500] if 'json_string_result' in locals() else 'N/A'; error_msg = f"PYTHON_ERROR: Ошибка парсинга JSON из DOM: {e}. Фрагмент: '{error_fragment}..."; logger.error(error_msg, exc_info=True); return {"error": error_msg}
        except Exception as e: error_msg = f"PYTHON_ERROR: Неожиданная ошибка при получении DOM: {e}"; logger.critical(error_msg, exc_info=True); return {"error": error_msg}

    async def wait_for_element_state(self, selector: str, state: str, timeout_seconds: int = 30):
        if not self.page or self.page.is_closed(): raise WaitTimeoutError(f"Не удалось ожидать '{state}' для '{selector}': страница закрыта.") # <<< Добавлено is_closed() check
        timeout_ms = timeout_seconds * 1000; logger.info(f"Ожидание состояния '{state}' для '{selector}' ({timeout_seconds}s)")
        playwright_state: Optional[Literal['visible', 'hidden', 'enabled', 'disabled', 'editable', 'checked']] = None; substring_to_check: Optional[str] = None; state_lower = state.lower().strip(); standard_states = ['visible', 'hidden', 'enabled', 'disabled', 'editable', 'checked']; text_match = re.match(r"text\((['\"]?)(.*?)\1\)$", state_lower)
        if text_match: substring_to_check = text_match.group(2); logger.debug(f"Ожидается текст: '{substring_to_check}'")
        elif state_lower in standard_states: playwright_state = state_lower; logger.debug(f"Ожидается Playwright состояние: '{playwright_state}'") # type: ignore
        else: raise ValueError(f"Неподдерживаемое состояние WAIT_FOR: '{state}'. Допустимы: {standard_states} или 'text(\"...\")'.")
        try:
            initial_find_timeout = int(timeout_ms * 0.9); locator = await self.find_element(selector, timeout=initial_find_timeout)
            if playwright_state:
                async def _wait_state_pw(): await locator.wait_for(state=playwright_state, timeout=timeout_ms)
                await self._retry_async(_wait_state_pw, timeout=timeout_ms + 1000); logger.info(f"Элемент '{selector}' достиг состояния '{playwright_state}'.")
            elif substring_to_check is not None:
                async def _wait_text_expect(): expect_timeout = max(1000, int(timeout_ms / (self.retry_attempts + 1))); await expect(locator).to_contain_text(substring_to_check, timeout=expect_timeout)
                await self._retry_async(_wait_text_expect, timeout=timeout_ms + 1000); logger.info(f"Элемент '{selector}' содержит текст '{substring_to_check}'.")
        except ValueError as e: logger.error(f"Ошибка WAIT_FOR: {e}"); raise BrowserInteractionError(f"Ошибка WAIT_FOR: {e}") from e
        except Exception as e:
            logger.error(f"Ошибка при ожидании '{state}' для '{selector}': {e}")
            if isinstance(e, (WaitTimeoutError, ElementNotFoundError)): raise e
            elif isinstance(e, PlaywrightTimeoutError): raise WaitTimeoutError(f"Таймаут ожидания '{state}' для '{selector}' (внутренний): {e}") from e
            else: raise BrowserInteractionError(f"Ошибка ожидания '{state}' для '{selector}': {e}") from e

    async def extract_data_from_element(self, selector: str, data_format: str = 'list_of_texts') -> List | List[List[str]] | List[Dict[str, str]]:
        if not self.page or self.page.is_closed(): raise BrowserInteractionError(f"Не удалось извлечь данные из '{selector}': страница закрыта.") # <<< Добавлено is_closed() check
        allowed_formats = ['list_of_texts', 'table_rows_as_lists', 'table_rows_as_dicts']
        if data_format not in allowed_formats: raise ValueError(f"Неподдерживаемый формат EXTRACT_DATA: '{data_format}'. Допустимы: {allowed_formats}")
        logger.info(f"Извлечение данных из '{selector}', формат: '{data_format}'"); effective_timeout = self.default_timeout
        script_list_of_texts = """(element) => { try { const texts = []; const walk = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null, false); let node; while(node = walk.nextNode()) { const parent = node.parentElement; if (parent && window.getComputedStyle(parent).display !== 'none' && parent.offsetParent !== null) { const text = node.nodeValue.trim(); if (text) texts.push(text); } } return JSON.stringify(texts); } catch(e) { return JSON.stringify({error: `JS Error (list_of_texts): ${e.message}`}); } }"""
        script_table_rows_as_lists = """(element) => { try { const rows = Array.from(element.querySelectorAll('tr')); const data = rows.map(row => Array.from(row.querySelectorAll('td, th')).map(cell => cell.textContent.trim()) ); return JSON.stringify(data); } catch(e) { return JSON.stringify({error: `JS Error (table_rows_as_lists): ${e.message}`}); } }"""
        script_table_rows_as_dicts = """(element) => { try { let headers = Array.from(element.querySelectorAll('thead th, thead td')).map(h => h.textContent.trim()); if (headers.length === 0) { headers = Array.from(element.querySelectorAll('tr:first-child th, tr:first-child td')).map(h => h.textContent.trim()); } const dataRows = Array.from(element.querySelectorAll(headers.length > 0 && element.querySelector('tbody') ? 'tbody tr' : 'tr:not(:first-child)')); const useIndicesAsKeys = headers.length === 0; const data = dataRows.map(row => { const rowData = {}; const cells = row.querySelectorAll('td'); Array.from(cells).forEach((cell, index) => { const key = useIndicesAsKeys ? `col_${index}` : (headers[index] || `col_${index}`); rowData[key] = cell.textContent.trim(); }); return Object.keys(rowData).length > 0 ? rowData : null; }).filter(row => row !== null); return JSON.stringify(data); } catch(e) { return JSON.stringify({error: `JS Error (table_rows_as_dicts): ${e.message}`}); } }"""
        script_map = {'list_of_texts': script_list_of_texts, 'table_rows_as_lists': script_table_rows_as_lists, 'table_rows_as_dicts': script_table_rows_as_dicts}
        script = script_map[data_format]
        try:
            container_locator = await self.find_element(selector, timeout=effective_timeout)
            logger.debug(f"Выполнение скрипта извлечения ({data_format}) для '{selector}'")
            json_result = await self._retry_async(lambda: container_locator.evaluate(script), timeout=effective_timeout)
            if not isinstance(json_result, str): raise BrowserInteractionError(f"Скрипт извлечения ({data_format}) не вернул JSON строку (тип: {type(json_result).__name__}).")
            extracted_data = json.loads(json_result)
            if isinstance(extracted_data, dict) and extracted_data.get('error'): js_error_msg = extracted_data['error']; logger.error(f"Ошибка JS при извлечении ({data_format}) из '{selector}': {js_error_msg}"); raise BrowserInteractionError(f"Ошибка JS скрипта ({data_format}): {js_error_msg}")
            count = len(extracted_data) if isinstance(extracted_data, list) else 'N/A'
            logger.info(f"Извлечено {count} записей ('{data_format}') из '{selector}'.")
            return extracted_data
        except ValueError as e: logger.error(f"Ошибка EXTRACT_DATA: {e}"); raise BrowserInteractionError(f"Ошибка EXTRACT_DATA: {e}") from e
        except json.JSONDecodeError as e: error_detail = f"Начало: '{str(json_result)[:200]}...'" if 'json_result' in locals() else "N/A"; logger.error(f"Ошибка JSON при извлечении ({data_format}) из '{selector}': {e}", exc_info=True); raise BrowserInteractionError(f"Ошибка JSON от скрипта ({data_format}). {error_detail}") from e
        except Exception as e:
            logger.error(f"Ошибка при извлечении ({data_format}) из '{selector}': {e}")
            if isinstance(e, (ElementNotFoundError, BrowserInteractionError)): raise e
            else: raise BrowserInteractionError(f"Не удалось извлечь ({data_format}) из '{selector}': {e}") from e

    async def press_key(self, selector: str, key: str, timeout: Optional[int] = None):
        if not self.page or self.page.is_closed(): raise BrowserInteractionError(f"Не удалось нажать '{key}' на '{selector}': страница закрыта.") # <<< Добавлено is_closed() check
        effective_timeout = timeout or self.default_timeout
        press_internal_timeout = int(effective_timeout * 0.8)
        async def _press_internal():
            element = await self.find_element(selector, timeout=effective_timeout)
            logger.info(f"Нажатие клавиши '{key}' на элементе: '{selector}'")
            await element.press(key, timeout=press_internal_timeout)
            logger.info(f"Клавиша '{key}' нажата на '{selector}'.")
        try: await self._retry_async(_press_internal, timeout=effective_timeout)
        except Exception as e:
            logger.error(f"Ошибка при нажатии '{key}' на '{selector}': {e}")
            if isinstance(e, (ElementNotFoundError, BrowserInteractionError)): raise e
            else: raise BrowserInteractionError(f"Нажатие '{key}' на '{selector}' не удалось: {e}") from e

    async def take_screenshot(self, path: str, full_page: bool = False, timeout: Optional[int] = None):
        if not self.page or self.page.is_closed(): logger.error("Не сделать скриншот: страница закрыта."); return # <<< Изменено: не бросаем ошибку, просто выходим
        effective_timeout = timeout or self.default_timeout
        logger.info(f"Снятие скриншота (full_page={full_page}) -> '{path}'...")
        try:
             async def _screenshot(): screenshot_timeout = max(15000, effective_timeout); await self.page.screenshot(path=path, full_page=full_page, timeout=screenshot_timeout)
             await self._retry_async(_screenshot, timeout=effective_timeout + 2000)
             logger.info(f"Скриншот сохранен: '{path}'.")
        except Exception as e: logger.error(f"Ошибка при снятии скриншота '{path}': {e}", exc_info=True)

    async def close_browser(self):
        logger.info("Начало закрытия браузера..."); closed_something = False
        if self.page and not self.page.is_closed():
             try: logger.debug("Закрытие страницы..."); await asyncio.wait_for(self.page.close(), timeout=10); logger.info("Страница закрыта."); closed_something = True
             except (Error, asyncio.TimeoutError) as e: logger.warning(f"Ошибка/таймаут закрытия страницы: {e}")
             finally: self.page = None
        if self.browser and self.browser.is_connected():
            try: logger.debug("Закрытие браузера..."); await asyncio.wait_for(self.browser.close(), timeout=20); logger.info("Браузер закрыт."); closed_something = True
            except (Error, asyncio.TimeoutError) as e: logger.warning(f"Ошибка/таймаут закрытия браузера: {e}")
            finally: self.browser = None
        if self.playwright:
             try: logger.debug("Остановка Playwright..."); loop = asyncio.get_running_loop(); await loop.run_in_executor(None, self.playwright.stop); logger.info("Playwright остановлен."); closed_something = True
             except Exception as e: logger.error(f"Ошибка остановки Playwright: {e}", exc_info=True)
             finally: self.playwright = None
        if not closed_something: logger.info("Браузер/Playwright уже были закрыты или не инициализированы.")
        logger.info("Закрытие браузера завершено.")

# --- END OF FILE browser_manager.py ---