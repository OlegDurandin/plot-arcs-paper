# Как воспроизвести результаты

Инструкция описывает, как из исходных корпусов получить все промежуточные данные и отчёты в `reports/`.
Ход работы, параметры и решения на каждом этапе записаны в `experiment_results.md`; план и решения ворот —
в `PLAN.md`.

## 1. Что нужно

- **Машина.** Проверено на macOS (Apple Silicon, MacBook M5, 24 ГБ). Нейросетевые шаги идут на `mps`; на другой
  платформе в конфигах поставьте `device: cpu` или `cuda` (будет медленнее или быстрее, результат классификаторов
  может отличаться в последних знаках). Свободного места — ≥ 40 ГБ: `data/processed` весит ≈ 17 ГБ, эмбеддинги полного
  корпуса добавляют ещё ≈ 4–5 ГБ.
- **Python 3.12** (использован 3.12.16).
- **Сеть** — для загрузки сказок Афанасьева (Викитека), лексиконов и моделей Hugging Face.
- **Ключ provod.ai** — только для шагов с LLM (раздел 5); стоимость указана там же.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # точная копия окружения (pip freeze)
# или минимальный набор с зафиксированными версиями:
# .venv/bin/pip install -r requirements.txt
export HF_HUB_DISABLE_XET=1                      # загрузка моделей HF обрывалась через Xet
```

Модели загружаются при первом запуске по ревизиям, зафиксированным в `configs/s02_emotion.yaml`:
`seara/rubert-tiny2-russian-sentiment` (ревизия `ad9be38671`), `cointegrated/rubert-tiny2-cedr-emotion-detection`
(`453ae93ca8`), `intfloat/multilingual-e5-small`; для предварительной локальной разметки —
`mlx-community/Qwen2.5-7B-Instruct-4bit`.

## 2. Данные

Корпуса в репозиторий не входят. Положите их в `data/raw/` (каталог только для чтения — код в него не пишет):

| файл | что это |
|---|---|
| `proza[2018].sqlite`, `proza[2019].sqlite`, `proza[2020-2021].sqlite`, `proza[2022-2023].sqlite`, `proza[2024-2025].sqlite`, `proza[2023,novel].sqlite`, `proza[2023,novella].sqlite`, `proza[2023,povest].sqlite`, `proza[2024,novel].sqlite`, `proza[2024,novella].sqlite`, `proza[2024,povest].sqlite`, `proza[2025,novel].sqlite`, `proza[2025,novella].sqlite`, `proza[2025,povest].sqlite` | Проза.ру |
| `fairytalez.sqlite` | fairytalez (машинный перевод) |
| `data_ficbook(24.11).zip` | Ficbook, ориджиналы (распаковывается кодом в `data/interim/ficbook_raw/`) |

Сказки Афанасьева (`s00c_afanasyev.py`) и лексиконы (`s02_fetch_lexicons.py`) скачиваются скриптами;
источники, лицензии и ссылки — в `configs/s00_inventory.yaml` и `configs/s02_emotion.yaml`.
LinisCrowd берётся из Wayback Machine (снимок 2022-04-01): оригинальный сайт больше не работает.

## 3. Общие правила

- Каждый скрипт запускается из корня репозитория: `.venv/bin/python src/<скрипт>.py configs/<конфиг>.yaml [опции]`.
- Все параметры — в `configs/*.yaml`, `seed: 42`.
- **Кэш.** Готовые результаты не пересчитываются. Этапы 0–1 кэшируются по хешу конфига и кода; остальные шаги
  пропускают файл, если он уже есть. Чтобы пересчитать шаг, удалите его выходной файл или каталог.
- **Выборка → полный корпус.** Большинство шагов сначала запускаются на выборке (`--sample` или без `--full`),
  затем на всём корпусе. Для воспроизведения итоговых цифр достаточно полных прогонов.
- **Путь к прогону этапа 1.** Каталог этапа 1 называется по хешу конфига и кода: `data/processed/s01_<hash>`.
  В репозитории он записан как `data/processed/s01_1700a4b18d` в `configs/s02_emotion.yaml`,
  `configs/s03_topics.yaml`, `configs/s04_nulls.yaml` и в `selection.stage1_exclusion` файла
  `configs/s00_inventory.yaml`. Если у вас получился другой хеш (другая версия кода или конфига), поправьте эти пути.

## 4. Этапы

Время указано там, где оно замерялось (MacBook M5).

### Этап 0. Инвентаризация и аналитическая выборка

```bash
.venv/bin/python src/s00c_afanasyev.py configs/s00_inventory.yaml       # Афанасьев из Викитеки
.venv/bin/python src/s00_inventory.py  configs/s00_inventory.yaml --full # манифест, длины, дубли (MinHash)
.venv/bin/python src/s00b_link_proza.py configs/s00_inventory.yaml      # склейка глав Проза.ру в серии
.venv/bin/python src/s00d_select.py    configs/s00_inventory.yaml       # analysis_set.parquet (первый проход)
.venv/bin/python src/s00_report.py     configs/s00_inventory.yaml --full # reports/00_inventory.md
```

Выход: `data/interim/manifest.parquet`, `data/interim/analysis_set.parquet`, `reports/00_inventory.md`,
`reports/00b_proza_linking_review.md`.

### Этап 1. Очистка, морфология, сегментация

```bash
.venv/bin/python src/s01_segment.py configs/s01_segment.yaml --full
.venv/bin/python src/s01_review.py  configs/s01_segment.yaml data/processed/s01_<hash>
.venv/bin/python src/s00d_select.py configs/s00_inventory.yaml          # второй проход: исключения по языку/орфографии этапа 1
```

`s00d_select.py` запускается второй раз, потому что исключение текстов не на русском и с дореформенной орфографией
опирается на `docs.parquet` этапа 1. Выход: `data/processed/s01_<hash>/` (предложения, сегменты N10/N20/N50 и окна W),
`reports/01_segmentation.md`, `reports/01_segmentation_review.md`.

### Этап 2. Тональность: сигналы и валидация

```bash
.venv/bin/python src/s02_fetch_lexicons.py configs/s02_emotion.yaml    # RuSentiLex, LinisCrowd, NRC + provenance.json
.venv/bin/python src/s02_lexicon.py  configs/s02_emotion.yaml --full    # лексиконные признаки сегментов N10/N20
.venv/bin/python src/s02_classify.py configs/s02_emotion.yaml --full    # классификаторы на MPS (≈ 4 ч)
.venv/bin/python src/s02_coverage.py configs/s02_emotion.yaml           # reports/02_coverage.md
.venv/bin/python src/s02_gold.py     configs/s02_emotion.yaml           # золото 400 + 30 повторов, reports/02_gold_annotation.html
```

Валидация по LLM (нужен ключ, раздел 5):

```bash
.venv/bin/python src/s02_llm_api.py configs/s02_llm_api.yaml --key gold --model deepseek-v4.1-flash
.venv/bin/python src/s02_llm_api.py configs/s02_llm_api.yaml --key gold --model gemini-2.5-flash-lite
.venv/bin/python src/s02_llm_api.py configs/s02_llm_api.yaml --key gold --model gemini-3.1-flash-lite
.venv/bin/python src/s02_llm_api.py configs/s02_llm_api.yaml --key gold --model gpt-5.4-nano
.venv/bin/python src/s02_llm_api.py configs/s02_llm_api.yaml --key gold --model gpt-5.4-mini
.venv/bin/python src/s02_llm_api.py configs/s02_llm_api.yaml --key gold --model gemini-3.5-flash --max-tokens 400
.venv/bin/python src/s02_silver.py   configs/s02_llm_api.yaml                    # серебро: 11 400 сегментов
.venv/bin/python src/s02_llm_api.py  configs/s02_llm_api.yaml --key silver       # deepseek-v4.1-flash
.venv/bin/python src/s02_validate_llm.py configs/s02_llm_api.yaml                # reports/02_validation_llm.md
.venv/bin/python src/s02_ensemble.py      configs/s02_ensemble.yaml              # reports/02_ensemble_position.md
.venv/bin/python src/s02_position_report.py configs/s02_ensemble.yaml            # reports/02_position_error.md
```

LLM-дуги (главный сигнал для сравнений со сказками):

```bash
.venv/bin/python src/s02_llm_arcs.py configs/s02_llm_arcs.yaml                   # выборка + оценка стоимости, без запросов
.venv/bin/python src/s02_llm_api.py  configs/s02_llm_arcs.yaml --key arcs --limit 2000   # проверка прогноза расхода
.venv/bin/python src/s02_llm_api.py  configs/s02_llm_arcs.yaml --key arcs        # остальное
.venv/bin/python src/s02_arcs_compare.py configs/s02_ensemble.yaml               # reports/02_llm_arcs.md
```

Необязательно: предварительная разметка золота локальной LLM (MLX) и её валидация —
`src/s02_llm_annotate.py configs/s02_emotion.yaml`, затем `src/s02_validate.py configs/s02_emotion.yaml --llm-only`
(`reports/02_validation_llm_preliminary.md`).

Человеческая разметка: откройте `reports/02_gold_annotation.html` в браузере (прогресс хранится в браузере), в конце —
«Выгрузить CSV»; затем `src/s02_validate.py configs/s02_emotion.yaml --gold <csv> --llm <csv LLM>`.

### Этап 3. Темы

```bash
.venv/bin/python src/s03_topics.py configs/s03_topics.yaml build
.venv/bin/python src/s03_topics.py configs/s03_topics.yaml train            # K = 50 и 100 (≈ 2 и 8 мин)
.venv/bin/python src/s03_topics.py configs/s03_topics.yaml infer --K 50     # θ сегментов N10/N20 (≈ 2 ч на K)
.venv/bin/python src/s03_topics.py configs/s03_topics.yaml traj  --K 50
.venv/bin/python src/s03_topics.py configs/s03_topics.yaml infer --K 100
.venv/bin/python src/s03_topics.py configs/s03_topics.yaml traj  --K 100
.venv/bin/python src/s03_report.py configs/s03_topics.yaml --K 50           # reports/03_topics_K50.md
.venv/bin/python src/s03_report.py configs/s03_topics.yaml --K 100
.venv/bin/python src/s03_robust.py configs/s03_robust.yaml embed            # эмбеддинги подвыборки
.venv/bin/python src/s03_robust.py configs/s03_robust.yaml report           # reports/03_robustness.md
.venv/bin/python src/s03_embed.py  configs/s03_robust.yaml --scope sample   # оценка времени на 2 % текстов
.venv/bin/python src/s03_embed.py  configs/s03_robust.yaml --scope full     # весь корпус N20 (фактически ≈ 19 ч), продолжает с места
# Этап 3s — Plot Arceology (субтитры Ororo и проза + фанфики); архив data/raw/Ororo(Final, 05.11.2026).zip,
# ororo.sqlite распакована из него в data/interim/ororo/. Сначала --sample, затем без флага (≈ 1 ч 20 мин всего).
for s in subs_ingest build train infer profiles; do .venv/bin/python src/s03s_arceology.py configs/s03s_arceology.yaml $s; done
# v2 (в статьях): дочистка ремарок и словаря субтитров. Перед полным прогоном скопировать прозу из v1:
#   data/processed/s03s/full/{prose,prose_meta.parquet,theta/prose_K*} → data/processed/s03s_v2/full/, models/s03s/full/lda_prose_K*.bin → models/s03s_v2/full/
for s in subs_ingest build train infer profiles; do .venv/bin/python src/s03s_arceology.py configs/s03s_arceology_v2.yaml $s; done
.venv/bin/python src/s03_embed_report.py configs/s03_robust.yaml --sample   # таблицы по ≈ 10 шардам
.venv/bin/python src/s03_embed_report.py configs/s03_robust.yaml            # reports/03_embeddings_full.md
```

**Недетерминизм LDA.** Обучение идёт в 8 потоков (`train.workers: 8`); tomotopy предупреждает, что при `workers ≠ 1`
результат может отличаться даже при фиксированном seed. Для точного повтора тем поставьте `workers: 1`
(обучение станет в несколько раз дольше); номера тем в отчётах (дискурсивные 8, 28, 34, 42 при K = 50)
относятся к модели из этого прогона.

### Этап 4. Нулевые модели, контроль длины, шум дуги

```bash
.venv/bin/python src/s04_nulls.py configs/s04_nulls.yaml arcs
.venv/bin/python src/s04_nulls.py configs/s04_nulls.yaml metrics     # 100 перестановок на текст (≈ 2 мин)
.venv/bin/python src/s04_nulls.py configs/s04_nulls.yaml noise       # бутстрэп предложений, 4 509 текстов, MPS
.venv/bin/python src/s04_nulls.py configs/s04_nulls.yaml report      # reports/04_nulls.md (MixedLM — долго, десятки минут)
.venv/bin/python src/s04_archetype_null.py configs/s04_nulls.yaml    # reports/04_archetype_null.md
.venv/bin/python src/s04_selection.py      configs/s04_nulls.yaml    # reports/04_selection_weights.md
.venv/bin/python src/s04_fanfic_endings.py configs/s04_nulls.yaml    # reports/04_fanfic_endings.md
```

Выборочный прогон — тот же порядок с `--sample` (`reports/04_nulls_sample.md`).

### Этап 5. RQ1 — средние дуги; концовка сказки (Ворота 5)

```bash
.venv/bin/python src/s05_mean_arcs.py configs/s05_mean_arcs.yaml                  # reports/05_mean_arcs.md
.venv/bin/python src/s05_ending.py configs/s05_ending.yaml --step contrast         # 2.1: концовка без последнего сегмента
.venv/bin/python src/s05_ending.py configs/s05_ending.yaml --step sample           # 2.2: ключ предложений + оценка стоимости
.venv/bin/python src/s02_llm_api.py configs/s05_ending.yaml --key sent --limit 50  # пробный прогон
.venv/bin/python src/s02_llm_api.py configs/s05_ending.yaml --key sent             # остальное (стоп 95 ₽)
.venv/bin/python src/s05_ending.py configs/s05_ending.yaml --step analyze          # 2.3–2.4: reports/05_ending.md
.venv/bin/python src/s05_ending_lex.py configs/s05_ending.yaml                   # правка v3, п. 4: классификатор и RSL на предложениях концовок
```

### Этап 6. RQ2 — расстояние до волшебной сказки

```bash
.venv/bin/python src/s06_distance.py configs/s06_distance.yaml                       # LLM-дуги: reports/06_distance.md
.venv/bin/python src/s06_distance.py configs/s06_distance.yaml --secondary --sample   # классификатор, 10 % корпуса
.venv/bin/python src/s06_distance.py configs/s06_distance.yaml --secondary            # классификатор, полный корпус
```

### Партия для студентов

```bash
.venv/bin/python src/s02_beginnings.py configs/s02_ensemble.yaml                 # reports/02_tale_beginnings.md
.venv/bin/python src/s02_students_batch.py configs/s02_students.yaml             # мощность + партия v2 (Ворота 5)
```

Отдаются `reports/02_students_batch1.html` (инструкция) и страницы разметчиков `reports/02_students_batch1_{A,B}.html`
(v2: 100 текстов на группу, 2 фрагмента середины). Устаревшие, не отдавать: `reports/02_students_batch1_v1*.html`
(120/1; `s02_students_batch.py configs/s02_students_batch1_v1.yaml`) и `reports/02_students_batch0.html`
(30 + 30; `s02_students_batch0.py`).

## 5. Шаги с LLM (provod.ai)

- Ключ — в файле `.env` в корне (в git не попадает):
  ```
  PROVOD_API_KEY=...
  PROVOD_BASE_URL=https://api.provod.ai/v1
  ```
- Стоимость берётся из ответа API (`usage.cost`, в рублях) и записывается в каждый CSV; при достижении `stop_at_rub`
  из конфига новые запросы не отправляются. Фактически потрачено: валидация (золото × 6 моделей + серебро) — 372 ₽;
  LLM-дуги — оценка ≈ 420 ₽ (стоп 520 ₽).
- Рассуждения моделей отключаются параметром `reasoning: {enabled: false}`; у gemini-3.5-flash и qwen3.8-flash
  они не отключаются (qwen не используется).
- **Повторный запуск не переразмечает готовое:** дописываются только недостающие `uid`.
- **Точного повтора ответов нет.** API при температуре 0 не детерминирован: DeepSeek на 30 скрытых повторах
  совпадает с собой по тональности в 90 % случаев, ρ = 0,95. Новый прогон даст близкие, но не те же самые
  цифры валидации. Ответы этого прогона хранятся в `data/processed/s02/llm_api/` и `data/processed/s02/llm_arcs/`
  с идентификатором модели и стоимостью; в репозиторий они не входят — для точного воспроизведения цифр отчётов
  их нужно получить у авторов.

## 6. Какой отчёт каким скриптом получен

| отчёт | скрипт |
|---|---|
| `reports/00_inventory.md`, `00b_proza_linking_review.md` | `s00_report.py`, `s00b_link_proza.py` |
| `reports/01_ficbook_months.csv` | разовый расчёт на Воротах 1 (скрипта нет; метод — в `experiment_results.md`, запись Ворот 1) |
| `reports/01_segmentation.md`, `01_segmentation_review.md` | `s01_review.py` |
| `reports/02_coverage.md` | `s02_coverage.py` |
| `reports/02_gold_annotation.html` | `s02_gold.py` |
| `reports/02_validation_llm_preliminary.md` | `s02_validate.py --llm-only` |
| `reports/02_validation_llm.md` | `s02_validate_llm.py` |
| `reports/02_ensemble_position.md` | `s02_ensemble.py` |
| `reports/02_position_error.md`, `02_position_error_by_form.csv`, `02_position_examples_*.csv` | `s02_position_report.py` |
| `reports/02_llm_arcs.md` | `s02_arcs_compare.py` |
| `reports/03_topics_K{50,100}.md`, `03_topics_sample_K20.md` | `s03_report.py` |
| `reports/03_robustness.md` | `s03_robust.py report` |
| `reports/04_nulls.md`, `04_nulls_sample.md` | `s04_nulls.py report` |
| `reports/04_archetype_null.md` | `s04_archetype_null.py` |
| `reports/04_selection_weights.md` | `s04_selection.py` |
| `reports/04_fanfic_endings.md` | `s04_fanfic_endings.py` |
| `reports/02_tale_beginnings.md` | `s02_beginnings.py` |
| `reports/02_students_batch1*.html`, `02_students_power.md` | `s02_students_batch.py` |
| `reports/05_mean_arcs.md` | `s05_mean_arcs.py` |
| `reports/05_ending.md` | `s05_ending.py` (шаги contrast → sample → разметка → analyze) |
| `reports/05_ending_lex.md` | `s05_ending_lex.py` (после разметки по предложениям; без LLM) |
| `reports/03s_arceology_v2[_sample].md`, `figures/03s_profiles_K{50,100,127}_v2.*` | `s03s_arceology.py` с `configs/s03s_arceology_v2.yaml` (основная версия в статьях) |
| `reports/03s_arceology[_sample].md`, `figures/03s_profiles_K127.*` | `s03s_arceology.py` (шаги subs_ingest → build → train → infer → profiles); LDA в 8 потоков — побитово не воспроизводится |
| `reports/06_distance.md`, `06_distance_secondary[_sample].md` | `s06_distance.py` |
| `reports/03_embeddings_full[_sample].md` | `s03_embed_report.py` |
| `reports/10_article_tables_figures.md` | вручную (черновик этапа 10) |
| `reports/10_figures.md`, `figures/10_fig*.{png,pdf}` | `s10_figures.py configs/s10_figures.yaml` |
| `reports/paper/ru/paper_ru.md`, `tables_ru.tex` | вручную (текст); счёт знаков — `s10_count.py reports/paper/ru/paper_ru.md` |

## 7. Анонимная копия для рецензирования

```bash
.venv/bin/python src/s10_anonymize.py configs/s10_anonymize.yaml   # → ../plot-arcs-paper-anon (вне репозитория)
```

Копия — из закоммиченного состояния; имена, e-mail, ссылки на репозиторий и абсолютные пути вычищаются, остаток
проверяется; создаётся новый git-репозиторий с одним коммитом от `Anonymous`. Загрузку на anonymous.4open.science
(или аналог) делает автор вручную. Пересобирать перед каждой подачей.

Рисунки — в `reports/figures/`, их имена начинаются с номера этапа.
