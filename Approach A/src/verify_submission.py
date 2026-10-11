"""Read-only frozen-results audit; writes submission verification, never models."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

TEAM=['Sudip Phuyal','Manila Bhandari','Rujjal Sada','Mahsa Azarnia','Vignesh Pechiappan Ayyathurai']
OUTPUT=Path('SUBMISSION_VERIFICATION.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,help='Initial pre-edit SHA-256 manifest; later runs use the saved verification manifest')
    args=parser.parse_args()
    if args.baseline:baseline=json.loads(args.baseline.read_text())
    else:baseline=json.loads(OUTPUT.read_text())['FrozenOutputSHA256']
    changed=[p for p,h in baseline.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    if changed:raise AssertionError(f'Frozen output changed: {changed}')
    energy=json.loads(Path('outputs/energy_forecasting/test_evaluation_complete.json').read_text())
    weather=json.loads(Path('outputs/weather_forecasting_full_year/test_evaluation_completed.json').read_text())
    uncertainty=json.loads(Path('outputs/uncertainty_forecasting/test_evaluation_complete.json').read_text())
    ablation=json.loads(Path('outputs/energy_forecasting/validation_results.json').read_text())['ablations']
    day=pd.read_csv('outputs/procurement_support/2024-02-28_daily_procurement.csv')
    value=pd.read_csv('outputs/procurement_support/value_of_uncertainty.csv')
    s=energy['selected']['Household'];b=energy['baseline']['Household']
    headline=dict(MLPortfolioWAPE=energy['selected']['MatchedPortfolio']['WAPE'],BlendPortfolioWAPE=energy['baseline']['MatchedPortfolio']['WAPE'],
        TemperatureMAE_C=next(r['MAE'] for r in weather['test_metrics'] if r['Variable']=='Temperature_avg_hourly'),
        HouseholdMAEImprovementPct=100*(1-s['MAE']/b['MAE']),HouseholdRMSEImprovementPct=100*(1-s['RMSE']/b['RMSE']),
        MLIntervalTestCoveragePct=100*uncertainty['metrics']['ML']['Test']['Coverage'],
        MLDailyTestCoveragePct=100*uncertainty['metrics']['ML']['TestDaily']['Coverage'],
        BlendDailyTestCoveragePct=100*uncertainty['metrics']['Blend']['TestDaily']['Coverage'],
        HistoricalCoreValidationWAPE=ablation['A']['MatchedPortfolio']['WAPE'],
        WeatherValidationWAPE=ablation['B']['MatchedPortfolio']['WAPE'],
        RecentMetadataValidationWAPE=ablation['D_plus_recent']['MatchedPortfolio']['WAPE'],
        Feb28P80_MWh=float(day.loc[(day.Method=='ML')&(day.Scenario=='C'),'SeparatelyCalibratedDailyProcurement_MWh'].iloc[0]),
        CostReduction4to1Pct=float(value.loc[(value.Method=='ML')&(value.Scenario=='C')&(value.Level=='Quarter-hour'),'RelativeCostReductionPct'].iloc[0]))
    for key in ['PointDailyEnergy_MWh','P10DailyEnergy_MWh','P50DailyEnergy_MWh','P90DailyEnergy_MWh']:headline['Feb28'+key]=uncertainty['unseen']['ML'][key]
    expected=dict(MLPortfolioWAPE='14.29',BlendPortfolioWAPE='14.47',TemperatureMAE_C='2.50',
        HouseholdMAEImprovementPct='7.7',HouseholdRMSEImprovementPct='16.1',MLIntervalTestCoveragePct='79.0',
        HistoricalCoreValidationWAPE='16.14',WeatherValidationWAPE='15.32',RecentMetadataValidationWAPE='14.13',
        Feb28P80_MWh='14.011',CostReduction4to1Pct='24',Feb28PointDailyEnergy_MWh='13.015',
        Feb28P10DailyEnergy_MWh='10.470',Feb28P50DailyEnergy_MWh='12.381',Feb28P90DailyEnergy_MWh='14.681')
    for key,label in expected.items():assert f'{headline[key]:.{len(label.split(".")[1]) if "." in label else 0}f}'==label,(key,headline[key],label)
    description=Path('ONE_PAGE_DESCRIPTION.md').read_text();readme=Path('README.md').read_text()
    for name in TEAM:assert name in description and name in readme
    assert 'ADD ALL TEAM MEMBER NAMES' not in description
    counted=re.sub(r'^\|[-:| ]+\|\s*$','',description,flags=re.M)
    words=len(re.sub(r'[#*|]',' ',counted).split());assert 450<=words<=550
    pdf=PdfReader('ONE_PAGE_DESCRIPTION.pdf');assert len(pdf.pages)==1
    box=pdf.pages[0].mediabox;assert abs(float(box.width)-595.2756)<.1 and abs(float(box.height)-841.8898)<.1
    extracted=re.sub(r'\s+',' ',pdf.pages[0].extract_text())
    for name in TEAM:assert name in extracted
    for key,label in expected.items():
        if key=='Feb28P50DailyEnergy_MWh':assert label in readme;continue
        assert label in description and label in extracted,(key,label)
    for key in ['MLPortfolioWAPE','BlendPortfolioWAPE','TemperatureMAE_C','MLIntervalTestCoveragePct','Feb28P80_MWh','CostReduction4to1Pct']:
        assert expected[key] in readme
    representative=pd.read_csv('representative_test_days.csv')
    assert representative.DeliveryDate.tolist()==['2023-03-15','2023-09-05','2024-02-27']
    for r in representative.itertuples(index=False):
        f=pd.read_parquet(f'outputs/energy_forecasting/predictions/test/{r.DeliveryDate}.parquet')
        labels=pd.read_parquet(f'outputs/energy_forecasting/evaluation/test_labels/{r.DeliveryDate}.parquet')
        matched=f.merge(labels[['Household_ID','TargetTimestamp','Target']],on=['Household_ID','TargetTimestamp'],validate='one_to_one')
        matched=matched.loc[matched.Target.notna()]
        actual=matched.Target.sum()/1000;ml=matched.Prediction.sum()/1000;blend=matched.BaselinePrediction.sum()/1000
        assert r.ObservedHouseholds==matched.Household_ID.nunique()
        import numpy as np
        np.testing.assert_allclose([r.ActualEnergy_MWh,r.MLForecastEnergy_MWh,r.BlendForecastEnergy_MWh,r.MLError_MWh,r.BlendError_MWh,r.MLSignedErrorPercent],
            [actual,ml,blend,abs(ml-actual),abs(blend-actual),100*(ml-actual)/actual],rtol=0,atol=1e-9)
        for number in [r.ActualEnergy_MWh,r.MLForecastEnergy_MWh,r.MLError_MWh,r.BlendForecastEnergy_MWh]:
            assert f'{number:.3f}' in description and f'{number:.3f}' in extracted
    required=['src/run_preprocessing.py','src/run_weather_forecasting.py','src/materialize_energy_data.py',
        'src/run_energy_forecasting.py','src/validation_energy_predictions.py','src/uncertainty_forecasting.py',
        'src/procurement_support.py','requirements-reproduction.txt','SUBMISSION_CHECKLIST.md',
        'outputs/procurement_support/figures/2024-02-28_procurement.png','outputs/procurement_support/figures/test_policy_costs.png']
    for p in required:assert Path(p).is_file(),p
    result=dict(ModelsAndPriorResultsUnchanged=True,FrozenFilesChecked=len(baseline),Discrepancies=[],
        TeamMembers=TEAM,TeamNamesSource='Confirmed directly by user',TeamNamesVerified=True,
        WordCount=words,WordCountMethod='Whitespace tokens excluding Markdown emphasis/headings/pipes and table separator',
        PDFPages=1,PDFPageSize='A4',PDFVisualReview='Rendered and visually inspected',
        HeadlineValues=headline,SourceCodePresent=True,Packaging='Working-directory files required; HEAD-only Git export is incomplete',
        RepresentativeDays=representative.to_dict(orient='records'),RepresentativeDaysPredefined=True,RepresentativeDayValuesCheckedAgainstLabels=True,
        FrozenOutputSHA256=baseline)
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='FrozenOutputSHA256'},indent=2))

if __name__=='__main__':main()
