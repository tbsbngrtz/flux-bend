@echo off
echo Starting 20 illustrative prompt sweep runs...
echo.
echo [%date% %time%] Running: prompt_21_victorian_botanical
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_21_victorian_botanical.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_21_victorian_botanical
if %errorlevel% neq 0 (
    echo ERROR on prompt_21_victorian_botanical - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_22_illuminated_manuscript
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_22_illuminated_manuscript.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_22_illuminated_manuscript
if %errorlevel% neq 0 (
    echo ERROR on prompt_22_illuminated_manuscript - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_23_ukiyo_e_crane_winter
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_23_ukiyo_e_crane_winter.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_23_ukiyo_e_crane_winter
if %errorlevel% neq 0 (
    echo ERROR on prompt_23_ukiyo_e_crane_winter - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_24_tarot_card_magician
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_24_tarot_card_magician.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_24_tarot_card_magician
if %errorlevel% neq 0 (
    echo ERROR on prompt_24_tarot_card_magician - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_25_fantasy_cartography
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_25_fantasy_cartography.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_25_fantasy_cartography
if %errorlevel% neq 0 (
    echo ERROR on prompt_25_fantasy_cartography - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_26_medieval_bestiary
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_26_medieval_bestiary.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_26_medieval_bestiary
if %errorlevel% neq 0 (
    echo ERROR on prompt_26_medieval_bestiary - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_27_art_nouveau_peacock
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_27_art_nouveau_peacock.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_27_art_nouveau_peacock
if %errorlevel% neq 0 (
    echo ERROR on prompt_27_art_nouveau_peacock - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_28_preraphealite_lake
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_28_preraphealite_lake.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_28_preraphealite_lake
if %errorlevel% neq 0 (
    echo ERROR on prompt_28_preraphealite_lake - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_29_deep_sea_scientific
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_29_deep_sea_scientific.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_29_deep_sea_scientific
if %errorlevel% neq 0 (
    echo ERROR on prompt_29_deep_sea_scientific - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_30_alchemical_diagram
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_30_alchemical_diagram.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_30_alchemical_diagram
if %errorlevel% neq 0 (
    echo ERROR on prompt_30_alchemical_diagram - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_31_folk_art_winter_village
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_31_folk_art_winter_village.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_31_folk_art_winter_village
if %errorlevel% neq 0 (
    echo ERROR on prompt_31_folk_art_winter_village - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_32_fantasy_dragon_hoard
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_32_fantasy_dragon_hoard.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_32_fantasy_dragon_hoard
if %errorlevel% neq 0 (
    echo ERROR on prompt_32_fantasy_dragon_hoard - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_33_vintage_travel_poster
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_33_vintage_travel_poster.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_33_vintage_travel_poster
if %errorlevel% neq 0 (
    echo ERROR on prompt_33_vintage_travel_poster - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_34_zentangle_cityscape
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_34_zentangle_cityscape.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_34_zentangle_cityscape
if %errorlevel% neq 0 (
    echo ERROR on prompt_34_zentangle_cityscape - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_35_celestial_map
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_35_celestial_map.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_35_celestial_map
if %errorlevel% neq 0 (
    echo ERROR on prompt_35_celestial_map - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_36_persian_miniature
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_36_persian_miniature.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_36_persian_miniature
if %errorlevel% neq 0 (
    echo ERROR on prompt_36_persian_miniature - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_37_stained_glass_cathedral
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_37_stained_glass_cathedral.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_37_stained_glass_cathedral
if %errorlevel% neq 0 (
    echo ERROR on prompt_37_stained_glass_cathedral - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_38_fairytale_forest_cottage
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_38_fairytale_forest_cottage.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_38_fairytale_forest_cottage
if %errorlevel% neq 0 (
    echo ERROR on prompt_38_fairytale_forest_cottage - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_39_heraldic_coat_of_arms
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_39_heraldic_coat_of_arms.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_39_heraldic_coat_of_arms
if %errorlevel% neq 0 (
    echo ERROR on prompt_39_heraldic_coat_of_arms - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo [%date% %time%] Running: prompt_40_ancient_egyptian_papyrus
call run.bat sweep --config configs/single_blockout_1_multi_prompt/block_dropout_single_block_1_prompt_40_ancient_egyptian_papyrus.yaml --out-dir outputs/single_blockout_1_multi_prompt/prompt_40_ancient_egyptian_papyrus
if %errorlevel% neq 0 (
    echo ERROR on prompt_40_ancient_egyptian_papyrus - errorlevel %errorlevel%
    echo Continuing to next...
)
echo.
echo.
echo All 20 illustrative sweeps completed.
pause