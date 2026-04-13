@echo off
echo Starting 40 block-5 dropout sweep runs...
echo.
echo [%date% %time%] Running: prompt_01_frog_suit
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_01_frog_suit.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_01_frog_suit
if %errorlevel% neq 0 (
    echo ERROR on prompt_01_frog_suit - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_02_red_sports_car
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_02_red_sports_car.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_02_red_sports_car
if %errorlevel% neq 0 (
    echo ERROR on prompt_02_red_sports_car - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_03_elderly_woman_portrait
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_03_elderly_woman_portrait.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_03_elderly_woman_portrait
if %errorlevel% neq 0 (
    echo ERROR on prompt_03_elderly_woman_portrait - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_04_golden_retriever_sunflowers
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_04_golden_retriever_sunflowers.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_04_golden_retriever_sunflowers
if %errorlevel% neq 0 (
    echo ERROR on prompt_04_golden_retriever_sunflowers - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_05_futuristic_city_sunset
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_05_futuristic_city_sunset.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_05_futuristic_city_sunset
if %errorlevel% neq 0 (
    echo ERROR on prompt_05_futuristic_city_sunset - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_06_red_apple_still_life
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_06_red_apple_still_life.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_06_red_apple_still_life
if %errorlevel% neq 0 (
    echo ERROR on prompt_06_red_apple_still_life - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_07_medieval_knight_forest
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_07_medieval_knight_forest.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_07_medieval_knight_forest
if %errorlevel% neq 0 (
    echo ERROR on prompt_07_medieval_knight_forest - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_08_abstract_geometric
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_08_abstract_geometric.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_08_abstract_geometric
if %errorlevel% neq 0 (
    echo ERROR on prompt_08_abstract_geometric - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_09_hello_world_sign
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_09_hello_world_sign.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_09_hello_world_sign
if %errorlevel% neq 0 (
    echo ERROR on prompt_09_hello_world_sign - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_10_butterfly_wing_macro
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_10_butterfly_wing_macro.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_10_butterfly_wing_macro
if %errorlevel% neq 0 (
    echo ERROR on prompt_10_butterfly_wing_macro - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_11_cat_on_books
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_11_cat_on_books.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_11_cat_on_books
if %errorlevel% neq 0 (
    echo ERROR on prompt_11_cat_on_books - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_12_coral_reef_underwater
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_12_coral_reef_underwater.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_12_coral_reef_underwater
if %errorlevel% neq 0 (
    echo ERROR on prompt_12_coral_reef_underwater - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_13_snow_mountain_dawn
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_13_snow_mountain_dawn.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_13_snow_mountain_dawn
if %errorlevel% neq 0 (
    echo ERROR on prompt_13_snow_mountain_dawn - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_14_steampunk_owl
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_14_steampunk_owl.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_14_steampunk_owl
if %errorlevel% neq 0 (
    echo ERROR on prompt_14_steampunk_owl - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_15_children_sandcastle_beach
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_15_children_sandcastle_beach.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_15_children_sandcastle_beach
if %errorlevel% neq 0 (
    echo ERROR on prompt_15_children_sandcastle_beach - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_16_ramen_overhead
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_16_ramen_overhead.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_16_ramen_overhead
if %errorlevel% neq 0 (
    echo ERROR on prompt_16_ramen_overhead - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_17_astronaut_space
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_17_astronaut_space.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_17_astronaut_space
if %errorlevel% neq 0 (
    echo ERROR on prompt_17_astronaut_space - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_18_art_nouveau_woman
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_18_art_nouveau_woman.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_18_art_nouveau_woman
if %errorlevel% neq 0 (
    echo ERROR on prompt_18_art_nouveau_woman - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_19_wolf_howling_moon
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_19_wolf_howling_moon.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_19_wolf_howling_moon
if %errorlevel% neq 0 (
    echo ERROR on prompt_19_wolf_howling_moon - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_20_isometric_japanese_garden
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_20_isometric_japanese_garden.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_20_isometric_japanese_garden
if %errorlevel% neq 0 (
    echo ERROR on prompt_20_isometric_japanese_garden - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_21_victorian_botanical
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_21_victorian_botanical.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_21_victorian_botanical
if %errorlevel% neq 0 (
    echo ERROR on prompt_21_victorian_botanical - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_22_illuminated_manuscript
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_22_illuminated_manuscript.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_22_illuminated_manuscript
if %errorlevel% neq 0 (
    echo ERROR on prompt_22_illuminated_manuscript - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_23_ukiyo_e_crane_winter
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_23_ukiyo_e_crane_winter.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_23_ukiyo_e_crane_winter
if %errorlevel% neq 0 (
    echo ERROR on prompt_23_ukiyo_e_crane_winter - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_24_tarot_card_magician
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_24_tarot_card_magician.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_24_tarot_card_magician
if %errorlevel% neq 0 (
    echo ERROR on prompt_24_tarot_card_magician - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_25_fantasy_cartography
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_25_fantasy_cartography.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_25_fantasy_cartography
if %errorlevel% neq 0 (
    echo ERROR on prompt_25_fantasy_cartography - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_26_medieval_bestiary
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_26_medieval_bestiary.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_26_medieval_bestiary
if %errorlevel% neq 0 (
    echo ERROR on prompt_26_medieval_bestiary - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_27_art_nouveau_peacock
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_27_art_nouveau_peacock.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_27_art_nouveau_peacock
if %errorlevel% neq 0 (
    echo ERROR on prompt_27_art_nouveau_peacock - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_28_preraphealite_lake
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_28_preraphealite_lake.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_28_preraphealite_lake
if %errorlevel% neq 0 (
    echo ERROR on prompt_28_preraphealite_lake - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_29_deep_sea_scientific
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_29_deep_sea_scientific.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_29_deep_sea_scientific
if %errorlevel% neq 0 (
    echo ERROR on prompt_29_deep_sea_scientific - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_30_alchemical_diagram
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_30_alchemical_diagram.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_30_alchemical_diagram
if %errorlevel% neq 0 (
    echo ERROR on prompt_30_alchemical_diagram - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_31_folk_art_winter_village
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_31_folk_art_winter_village.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_31_folk_art_winter_village
if %errorlevel% neq 0 (
    echo ERROR on prompt_31_folk_art_winter_village - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_32_fantasy_dragon_hoard
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_32_fantasy_dragon_hoard.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_32_fantasy_dragon_hoard
if %errorlevel% neq 0 (
    echo ERROR on prompt_32_fantasy_dragon_hoard - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_33_vintage_travel_poster
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_33_vintage_travel_poster.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_33_vintage_travel_poster
if %errorlevel% neq 0 (
    echo ERROR on prompt_33_vintage_travel_poster - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_34_zentangle_cityscape
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_34_zentangle_cityscape.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_34_zentangle_cityscape
if %errorlevel% neq 0 (
    echo ERROR on prompt_34_zentangle_cityscape - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_35_celestial_map
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_35_celestial_map.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_35_celestial_map
if %errorlevel% neq 0 (
    echo ERROR on prompt_35_celestial_map - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_36_persian_miniature
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_36_persian_miniature.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_36_persian_miniature
if %errorlevel% neq 0 (
    echo ERROR on prompt_36_persian_miniature - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_37_stained_glass_cathedral
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_37_stained_glass_cathedral.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_37_stained_glass_cathedral
if %errorlevel% neq 0 (
    echo ERROR on prompt_37_stained_glass_cathedral - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_38_fairytale_forest_cottage
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_38_fairytale_forest_cottage.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_38_fairytale_forest_cottage
if %errorlevel% neq 0 (
    echo ERROR on prompt_38_fairytale_forest_cottage - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_39_heraldic_coat_of_arms
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_39_heraldic_coat_of_arms.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_39_heraldic_coat_of_arms
if %errorlevel% neq 0 (
    echo ERROR on prompt_39_heraldic_coat_of_arms - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_40_ancient_egyptian_papyrus
call run.bat sweep --config configs/single_blockout_5_multi_prompt/block_dropout_single_block_5_prompt_40_ancient_egyptian_papyrus.yaml --out-dir outputs/single_blockout_5_multi_prompt/prompt_40_ancient_egyptian_papyrus
if %errorlevel% neq 0 (
    echo ERROR on prompt_40_ancient_egyptian_papyrus - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo.
echo All 40 block-5 sweeps completed.
pause