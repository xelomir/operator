# --- START OF FILE README.md ---
# AI Operator - Автоматизация Задач в Браузере

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI Operator** - это Python-приложение, использующее AI (в данный момент Groq API с Llama3) для управления браузером (через Playwright) с целью выполнения задач, описанных на естественном языке.

## Основные Возможности

*   **Управление Браузером:** Использует Playwright для навигации, кликов, ввода текста, скроллинга и извлечения данных со страниц.
*   **AI Агент:** Взаимодействует с Groq API для принятия решений о следующих шагах на основе цели и текущего состояния DOM.
*   **Конфигурация:** Настройки оператора, браузера и агента вынесены в файл `config.yaml`.
*   **Механизм Reflection:** Позволяет агенту анализировать свои ошибки и корректировать план действий.
*   **Summarization:** Периодически сокращает историю и план для поддержания контекста.
*   **Обработка Ошибок:** Включает механизмы повторных попыток (retry) и восстановления (Reload+Retry).
*   **CBR (Case-Based Reasoning):** Экспериментальная база данных для оценки эффективности селекторов и их приоритезации.
*   **Настраиваемое Логирование:** Использует `rich` для красивого вывода в консоль и записывает подробные DEBUG логи в файл.
*   **Управление Зависимостями:** Список зависимостей в `requirements.txt`.

## Установка и Настройка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone <URL_вашего_репозитория>
    cd <папка_репозитория>
    ```

2.  **Создайте виртуальное окружение (рекомендуется):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # или
    .\venv\Scripts\activate  # Windows
    ```

3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Установите браузеры для Playwright (если еще не установлены):**
    ```bash
    playwright install chromium  # Или другие: firefox, webkit
    ```

5.  **Настройте API Ключ:**
    *   Создайте файл `.env` в корневой папке проекта.
    *   Добавьте в него ваш Groq API ключ:
        ```dotenv
        GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```
    *   Либо передавайте ключ через аргумент командной строки `--groq-api-key`.

6.  **Настройте `config.yaml` (опционально):**
    *   Откройте файл `config.yaml`.
    *   Измените параметры по необходимости (например, модель Groq, таймауты, уровень лога и т.д.).

## Использование

Запустите оператора из командной строки, указав цель:

```bash
python operator_core.py "Ваша цель здесь"

Примеры:

Зайти на сайт и найти информацию:

python operator_core.py "Открыть сайт https://ru.wikipedia.org/ и найти год основания Python"
Use code with caution.
Bash
Запустить с определенным URL и в режиме без GUI:

python operator_core.py "Найти заголовок на странице" -u "example.com" --headless
Use code with caution.
Bash
Использовать другой файл конфигурации:

python operator_core.py "Моя цель" --config my_config.yaml
Use code with caution.
Bash
Переопределить уровень лога:

python operator_core.py "Отладить задачу" -l DEBUG
Use code with caution.
Bash
Оператор будет выполнять шаги, логируя свои действия в консоль и в файл журнала (путь к файлу указывается при запуске).

Лицензия
Этот проект лицензирован под лицензией MIT - см. файл LICENSE для подробностей.

TODO / Идеи для Развития
Добавление поддержки других LLM (OpenAI, Claude).

Улучшение механизма Reflection и Summarization.

Более продвинутые стратегии выбора селекторов (визуальные?).

Интеграция мультимодальных моделей для анализа скриншотов.

Структурированное представление плана.

Больше Unit и интеграционных тестов.

--- END OF FILE README.md ---
---

**2. `LICENSE`** (MIT License)

```text
# --- START OF FILE LICENSE ---
MIT License

Copyright (c) 2025 xelomir

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
