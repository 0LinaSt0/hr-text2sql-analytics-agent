add_lang_features_promt = """Правильные способы обработки вопросов по языкам:
        1)Вопрос: Сколько человек в банке знает немецкий?
            SQL output: 
                SELECT count(distinct employee_id) AS german_speakers
                FROM {TABNAME}
                WHERE arrayExists(x -> x ILIKE '%немецкий%', lang_with_level)

        2)Вопрос: Выведи фамилии сотрудников которые свободно говорят на английском и знают итальянский?
            SQL output: 
                SELECT person_surname
                FROM {TABNAME}
                WHERE (arrayExists(x -> x ILIKE '%английский% proficiency', lang_with_level)
                    OR 
                    arrayExists(x -> x ILIKE '%английский% advanced', lang_with_level))
                    and
                    arrayExists(x -> x ILIKE '%итальянский%', lang_with_level)

        3)Вопрос: Покажи имена, фамилии, грейд и уровень знания английского сотрудников?
            SQL output: 
                SELECT person_name, person_surname, grade_num,
                        trim(splitByString('/', arrayFirst(x -> x ILIKE '%английский%', lang_with_level))[2]) as english_level
                    FROM {TABNAME}
                    WHERE arrayExists(x -> x ILIKE '%английский%', lang_with_level)
        
        4)Вопрос: Кто владеет французским на высоком уровне и старше 40 лет?
            SQL output: 
                SELECT employee_id, person_name, person_surname, age_y
                FROM {TABNAME}
                WHERE (arrayExists(x -> x ILIKE '%французский% proficiency', lang_with_level)
                    OR 
                    arrayExists(x -> x ILIKE '%французский% advanced', lang_with_level)
                    OR
                    arrayExists(x -> x ILIKE '%французский% upper-intermediate', lang_with_level)) and age_y > 40

        5)Вопрос: Табельный сотрудников кто не указал знание языков
            SQL output: 
                SELECT employee_id, lang_with_level
                FROM {TABNAME}
                WHERE arrayAll(x -> x = 'Не заполнено', lang_with_level)

        6)Вопрос: Покажи распределение по знанию языков в компании
            SQL output: 
                SELECT
                    trim(splitByString('/', lang_item)[1]) as language,
                    COUNT(*) as employee_count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT employee_id) FROM {TABNAME}), 2) as percentage
                FROM {TABNAME}
                ARRAY JOIN lang_with_level as lang_item
                WHERE lang_item != ''
                GROUP BY trim(splitByString('/', lang_item)[1])
                ORDER BY employee_count DESC
        
        7)Вопрос: Сколько сотрудников владеют языками на высоком уровне и какими?
            SQL output: 
                SELECT DISTINCT
                    trim(splitByString('/', item)[1]) AS language,
                    count(*) as count_employees
                FROM {TABNAME}
                ARRAY JOIN lang_with_level AS item
                WHERE lower(trim(splitByString('/', item)[2])) IN ('advanced', 'proficiency')
                GROUP BY trim(splitByString('/', item)[1])
                ORDER BY count_employees DESC
        
        8)Вопрос: Покажи распределение сотрудников по уровню владения английским языком.
            SQL output:
                SELECT 
            lower(trim(splitByString('/', arrayFirst(x -> x ILIKE '%английский%', lang_with_level))[2])) as english_level,
            count(distinct employee_id) as employee_count
        FROM {TABNAME}
        WHERE arrayExists(x -> x ILIKE '%английский%', lang_with_level)
            AND lower(trim(splitByString('/', arrayFirst(x -> x ILIKE '%английский%', lang_with_level))[2])) IN 
                ('elementary', 'pre-intermediate', 'intermediate', 'upper-intermediate', 'advanced', 'proficiency')
        GROUP BY english_level
        ORDER BY 
            CASE english_level
                WHEN 'elementary' THEN 1
                WHEN 'pre-intermediate' THEN 2
                WHEN 'intermediate' THEN 3
                WHEN 'upper-intermediate' THEN 4
                WHEN 'advanced' THEN 5
                WHEN 'proficiency' THEN 6
            END

        Уровни знания языков в таблице по возрастанию: 'elementary', 'pre-intermediate', 'intermediate', 'upper-intermediate', 'advanced', 'proficiency'.
        Используй для уровня владения только эти значения.
"""

add_find_boss_promt = """\n· Если пользователь хочет узнать/найти **своего руководителя**(в чьем подчинении, команде работает) используй запрос:
        WITH my_boss AS (SELECT  COALESCE(
            NULLIF(NULLIF(po_i_pernr, 0), employee_id),
            NULLIF(NULLIF(lid_cluster_i_pernr, 0), employee_id),
            NULLIF(NULLIF(it_lid_cluster_i_pernr, 0), employee_id),
            NULLIF(NULLIF(lid_tribe_i_pernr, 0), employee_id),
            NULLIF(NULLIF(cur_tribe_i_pernr, 0), employee_id),
            NULLIF(NULLIF(lid_3_lvl_i_pernr, 0), employee_id),
            NULLIF(NULLIF(lid_2_lvl_i_pernr, 0), employee_id),
            NULLIF(NULLIF(lid_1_lvl_i_pernr, 0), employee_id)
            ) AS my_boss_id
        FROM {TABNAME}
        WHERE employee_id = {user_id} and report_date = '{ACTUAL_DATA}')

        SELECT employee_id, person_surname, person_name, person_patronimics
        FROM {TABNAME}
        WHERE employee_id = (SELECT my_boss_id FROM my_boss) --корректный фильтр
              and report_date = '{ACTUAL_DATA}'
(ВАЖНО: в случае поиска руководителя **другого конкретного сотрудника** необходимо использовать employee_id этого сотрудника в фильтре WHERE в CTE my_boss)"""