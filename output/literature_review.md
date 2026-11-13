# Literature Review: Demographics and US Healthcare Expenditures

*Prepared for ECON 442 research project (MEPS 2022, N = 17,387 adults, survey-weighted).*

**How the citations were checked.** Each entry is tagged:
- **[V]**: I am confident of the authors, year, venue and main finding from well-established knowledge of the literature.
- **[V~]**: the citation is right, but the exact number is approximate or from memory. Check it against the PDF before quoting.
- **[U]**: unverified, or the citation as given in the prompt seems to be wrong. A correction is suggested.

No live web search was done for this draft. Before submitting, check every number marked [V~] or [U] against the original paper.

---

## 1) Literature summary table

| # | Study | Data | Method | Key finding | Status |
|---|---|---|---|---|---|
| **(a) Modeling spending and how concentrated it is** |||||
| 1 | Duan, Manning, Morris & Newhouse (1983), *J. Business & Economic Statistics* 1(2):115–126 | RAND Health Insurance Experiment (HIE) | Four-equation / two-part model (any use × log positive spending) | Two-part models fit the mass of zeros and the skewed positive spending better than one-equation Tobit or sample-selection models. | [V] |
| 2 | Duan (1983), *JASA* 78:605–610 | — | "Smearing" retransformation | Gives a consistent estimate of E[y] from log-scale OLS without assuming normal errors. Biased if errors are heteroskedastic. | [V] |
| 3 | Manning, Newhouse, Duan, Keeler, Leibowitz & Marquis (1987), *American Economic Review* 77(3):251–277 | RAND HIE (about 2,000 families, random assignment of coinsurance) | Randomized experiment, two-part model | Free care raised spending about 30% (roughly 40% in some specifications) vs. the 95% coinsurance plan. Demand elasticity is about −0.2. | [V~] |
| 4 | Manning & Mullahy (2001), *J. Health Economics* 20(4):461–494 | Simulations | Log-OLS vs. GLM (log link, gamma/Poisson) | No single estimator wins. Use a modified Park test and kurtosis of log residuals to choose. Log-OLS is more efficient when the log-scale residuals are heavy-tailed. GLM is better under heteroskedasticity. | [V] |
| 5 | Buntin & Zaslavsky (2004), *J. Health Economics* 23(3):525–542 | Medicare Current Beneficiary Survey | Eight estimators compared (OLS, log, square root, GLM, two-part) | Two-part GLM / OLS models give similar out-of-sample accuracy. Model choice matters less than the "ado" suggests. Use cross-validated diagnostics. | [V] |
| 6 | AHRQ MEPS Statistical Briefs on concentration (Cohen & Yu, #354, 2012 for 2009 data; Mitchell, #497, 2016; later briefs, e.g., #533) | MEPS-HC | Weighted percentile shares | **Top 1% ≈ 20–23% of spending. Top 5% ≈ 48–50%. Top 10% ≈ 63–66%. Bottom 50% ≈ 3%.** Very stable across years. | [V~] (brief numbers and exact percentages should be checked) |
| 7 | Berk & Monheit (1992; 2001), *Health Affairs* | NMES/MEPS | Concentration curves | Spending has been heavily concentrated since the 1970s. Top 1% ≈ 27–28% in 1996 data. | [V~] |
| **(b) Racial/ethnic disparities** |||||
| 8 | Dieleman et al. (2021), *JAMA* 326(7):649–659, "US Health Care Spending by Race and Ethnicity, 2002–2016" | MEPS + other sources (Disease Expenditure Project) | Age-standardized per-capita spending | White per-capita spending is above the all-population mean. Hispanic and Asian/Pacific Islander (API) spending is well below it. Black spending is near average overall, with more ED/inpatient use and less ambulatory/dental use. | [V~] |
| 9 | Dickman, Gaffney, McGregor, Himmelstein, McCormick, Bor & Woolhandler (2022), *JAMA Network Open* 5(4):e227403, "Trends in Health Care Use Among Black and White Persons in the US, 1963–2019" | MEPS and predecessor surveys | Per-capita spending/use trends | Per-capita spending for Black people has been lower than for White people for decades (about 26–30% lower in recent years). **The prompt's "Dickman et al. 2022 *JAMA*" is most likely this *JAMA Netw Open* paper.** The Black/Hispanic/Asian < White pattern is also in Dieleman 2021. | [U] (venue corrected; check the exact %) |
| 10 | McGuire, Alegría, Cook, Wells & Zaslavsky (2006), *Health Services Research* 41(5):1979–2005 | MEPS | IOM definition implemented with rank-and-replace | IOM disparity = all differences except those due to clinical need/preferences. Adjust for health status, but let SES and insurance differences count as part of the disparity. | [V] |
| 11 | Cook, McGuire & Zaslavsky (2012), *Health Services Research* 47(3 Pt 2):1232–1254 | MEPS | Methods review (rank-and-replace, propensity score, Peters–Belson) | Different definitions give different disparity sizes. Conditioning on SES understates the IOM disparity. | [V] |
| 12 | Cook, McGuire, Lock & Zaslavsky (2010), *Health Services Research* 45(3) | MEPS | IOM-concordant estimates of expenditure disparities | Black and Hispanic adults have substantially lower total expenditures than White adults after adjusting for need. | [V~] |
| 13 | Mahajan, Caraballo, Lu, Valero-Elizondo, Massey, Annapureddy, … Krumholz (2021), *JAMA* 326(7):637–648 | NHIS 1999–2018 | Trend analysis | Black–White and Hispanic–White gaps in self-rated health, insurance and cost-related unmet need narrowed after the ACA but remained. Low-income Black/Hispanic adults are the worst off. (This is about access and affordability, not spending.) | [V~] |
| **(c) Income, insurance and out-of-pocket (OOP) burden** |||||
| 14 | Banthin & Bernard (2006), *JAMA* 296(22):2712–2719 | MEPS 1996–2003, ages 18–64 | OOP (incl. premiums) / family income; thresholds of 10% and 20% | The share with high burden rose from 15.8% to 19.2% (more than 10% of income) between 1996 and 2003. Burden is concentrated among the poor/low income, the publicly insured and the chronically ill. | [V~] |
| 15 | Banthin, Cunningham & Bernard (2008), *Health Affairs* 27(1):188–195 | MEPS 1996–2004 | Same | Financial burden rose fastest for low-income and privately insured people. | [V~] |
| 16 | Hadley, Holahan, Coughlin & Miller (2008), *Health Affairs* 27(5):w399–w415 | MEPS | Accounting | The uninsured use about half as much care per capita as the insured. Uncompensated care is about $56B. | [V~] |
| 17 | Cunningham (2010+); Collins et al. (Commonwealth Fund) | MEPS/CPS | Descriptive | Underinsurance and OOP burden are rising among the privately insured with high deductibles. | [V~] |
| **(d) Machine-learning cost prediction and risk adjustment** |||||
| 18 | Pope, Kautter, Ellis, Ash, Ayanian, Iezzoni et al. (2004), *Health Care Financing Review* 25(4):119–141 | Medicare 5% sample | CMS-HCC prospective model | **Prospective R² ≈ 0.11–0.12 (dollars). A demographic-only model gets ≈ 0.01.** Later CMS-HCC versions report ≈ 0.12–0.13. | [V~] |
| 19 | van de Ven & Ellis (2000), *Handbook of Health Economics* Vol. 1A, ch. 14 | Review | Risk adjustment | Age/sex-only R² ≈ 0.01–0.03. Diagnosis-based prospective ≈ 0.10–0.15. Concurrent ≈ 0.30–0.50. Adding prior cost gives prospective ≈ 0.15–0.25. | [V] |
| 20 | Rose (2016), "A machine learning framework for plan payment risk adjustment," *Health Services Research* 51(6):2358–2374. The prompt's "*Biometrics*" venue could not be confirmed | Truven MarketScan (subsample) | Super Learner ensemble vs. parametric | A Super Learner using only about 10 variables kept about 92% of the full-model R². The ensemble gives a small gain over OLS. Reported cross-validated R² values are in the ~0.1–0.25 range (check). | [U] venue / [V~] finding |
| 21 | Shrestha, Bergquist, Montz & Rose (2018), *Health Services Research* 53(S1) / Park & Basu (2018) | MarketScan / Medicare | ML for risk adjustment | ML gives modest R² gains over the HCC-OLS approach. Most of the signal is in diagnoses, not in how flexible the functional form is. | [V~] |
| 22 | Kan, Kharrazi, Chang, Bodycombe, Lemke & Weiner (2019), *PLoS ONE* 14(3):e0213258 | Commercial claims (ACG) | OLS vs. LASSO/ridge/elastic net | Penalized regression does no better than OLS on cost R². **The prompt's "Kan, Ellis et al." could not be matched. The closest is this Kan, Weiner et al. paper. Ellis's ML work: Ellis, Martins & Rose (2018), in McGuire & van Kleef (eds.), *Risk Adjustment, Risk Sharing and Premium Regulation in Health Insurance Markets*, Academic Press.** | [U] |
| 23 | Morid, Kawamoto, Ault, Dorius & Abdelrahman (2017), *AMIA Annual Symposium Proceedings* 2017:1312–1321 | Systematic review + empirical test (Utah claims) | Supervised learning review | Prior cost is the strongest predictor. GBM/ensembles do best. Cost-bucket prediction is common. | [V~] |
| 24 | Yang, Delcher, Shenkman & Ranka (2018), *BioMedical Engineering OnLine* 17(Suppl 1):131 | Florida Medicaid / commercial claims | Linear, LASSO, regression trees, GBM | Predicting high-need high-cost patients: GBM is best. Prior-period cost and utilization features dominate. | [V~] |
| 25 | Bertsimas, Bjarnadóttir, Kane, Kryder, Pandey, Vempala & Wang (2008), *Operations Research* 56(6):1382–1392 | Commercial claims, about 800K members | CART + clustering on cost buckets | ML with cost-history features beats baseline. The most useful information is the recent cost trajectory. Medical (diagnosis) data add only a little on top of cost history. | [V] |
| 26 | Obermeyer, Powers, Vogeli & Mullainathan (2019), *Science* 366(6464):447–453 | Academic health system, about 50K patients | Audit of a commercial risk algorithm | At the same risk score, Black patients had about 26% more chronic illnesses. Using cost as the label builds in lower spending on Black patients. Re-labeling would raise the Black share of auto-enrolled patients from 17.7% to 46.5%. | [V] |
| **(e) Segmenting patients by cost or need** |||||
| 27 | Joynt Figueroa, Zhou, Jha & Orav (2017), *Healthcare* (Amsterdam) 5(1–2):62–67 ("Segmenting high-cost Medicare patients into potentially actionable cohorts"); Joynt et al. (2013), *Am J Managed Care* | Medicare FFS | Rule-based segmentation | Six segments (e.g., frail elderly, major complex chronic, under-65 disabled, ESRD). Frail elderly + major complex chronic make up most high-cost spending. | [V~] |
| 28 | Hayes, Salzberg, McCarthy, Radley, Abrams, Shah & Anderson (2016), Commonwealth Fund Issue Brief | MEPS 2009–2011 | Need definition (3+ chronic conditions + functional limitation) | High-need adults are about 5% of adults. Their per-capita spending is about $21K, roughly 4 times average. Their OOP is higher too. | [V~] |
| 29 | Blumenthal, Chernof, Fulmer, Lumpkin & Selberg (2016), *NEJM* 375:909–911 | Commentary | — | Framework for high-need high-cost care. | [V] |
| 30 | Latent-class / cluster studies, e.g., Lafortune et al.; Yan, Liu et al. (2019) high-cost cluster studies in *Health Services Research*/*BMJ Open* | Various | LCA, k-means | Usually find 4–8 classes, with a multimorbid elderly class and a "healthy low-use" class. **Specific citations not verified. Do not cite them without checking.** | [U] |
| **(f) Double/debiased machine learning (DML) in health economics** |||||
| 31 | Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018), *Econometrics Journal* 21(1):C1–C68 | Theory | DML: Neyman-orthogonal scores + cross-fitting | √n-consistent inference on a low-dimensional parameter with ML nuisance functions. | [V] |
| 32 | Kreif & DiazOrdaz (2019), *Oxford Research Encyclopedia of Economics and Finance* | Review | ML for causal inference in health | Reviews TMLE/DML for health-policy evaluation. | [V~] |
| 33 | Applications (e.g., Knaus 2022, *Econometrics J*; DML for effects of insurance/Medicaid on spending) | Various | DML / causal forests | DML is increasingly used. Few applications are to MEPS racial/income spending gaps. **No specific MEPS–DML disparity paper verified.** | [U] |

---

## 2) How our results compare

| Our result (MEPS 2022) | Literature benchmark | Verdict |
|---|---|---|
| Mean total spending $7,762 per adult | AHRQ MEPS adult means for 2019–2021 are about $7,000–8,000 | **Consistent** |
| Top 5% = 46.5%, top 1% = 19.5% | AHRQ: top 5% ≈ 48–50%, top 1% ≈ 20–23% | **Consistent, slightly less concentrated.** Plausible because the sample is adults only (excluding children reduces the zeros, which lowers concentration) and because of 2022 post-pandemic utilization. Worth a sentence. |
| NH White $9,119 > Black $6,540 > Asian $5,553 > Hispanic $4,618 | Dieleman 2021; Dickman 2022: White highest, Hispanic/Asian lowest, Black in between and lower than White | **Consistent.** Our Black/White ratio (0.72) is close to the ~26–30% gap in Dickman et al. These are unadjusted gaps. The IOM approach (McGuire/Cook) would adjust for age and health status only. |
| Uninsured $1,278 vs. public-only $10,296 | Hadley et al. 2008: uninsured use about half as much as the insured | **Consistent in direction, larger in size.** Our public-only group is weighted toward Medicare (older, sicker), so composition explains much of the gap. |
| Near-poor $11,349 > high income $8,001 | Lower-income people are sicker, so spending is flat or U-shaped by income once Medicaid/Medicare are included | **Consistent** (the gradient by need). |
| OOP highest for high income ($1,361), lowest for poor ($500) | Banthin & Bernard: OOP *dollars* rise with income, but the OOP *share of income* is highest for the poor | **Consistent.** Report OOP/income as well, as the clustering does. |
| Older-chronic cluster: 18% of the population, $16,476, OOP/income 9.6% vs. 2.5% for the healthy-affluent cluster | Hayes 2016 high-need ≈ 5% at ≈ $21K; Joynt Figueroa frail/complex segments | **Consistent but broader.** Our cluster is larger and less extreme because K-means on demographics produces broad groups, not a definition based on need. 9.6% is close to the 10% "high burden" threshold used by Banthin & Bernard. |
| Young Hispanic, often-uninsured cluster: $2,892, 64% with any use | Many zeros among the uninsured/Hispanic; this motivates two-part models | **Consistent** |
| R² (log) 0.35 OLS / 0.37 KNN; R² ≈ 0 in dollars | Demographic-only dollar R² ≈ 0.01–0.03. CMS-HCC prospective ≈ 0.12. Concurrent diagnosis models ≈ 0.3–0.5 | **Partly differs.** A log-scale R² of 0.35 is not comparable to the dollar R² in risk-adjustment work. Our dollar R² of about 0 is **lower** than even the demographic-only benchmark. This may be because health-status variables are self-reported (not diagnosis-based) and because of extreme tails in the test split. It is also concurrent, cross-sectional data. Check smearing/retransformation, and consider a two-part GLM. |
| Top-10% AUC 0.78 (KNN) vs. 0.79 (logit) | High-cost classification AUCs ≈ 0.75–0.85 when claims history is available (Yang 2018; Morid 2017) | **Consistent.** It is notable that we reach this level without utilization or prior cost. ML ≈ logit matches Kan 2019 and Buntin & Zaslavsky (flexibility adds little). |

---

## 3) What this project adds (candidate contributions, phrased honestly)

1. **Updated descriptive benchmarks for 2022, after the pandemic.** Most published concentration and race/ethnicity spending figures use data from before 2020. Ours recompute them for adults in 2022. *Novel only as an update. AHRQ publishes similar tables each year. Novel within the scope of a class project.*
2. **Unsupervised profiles that combine demographics with OOP burden.** The usual segmentation (Joynt Figueroa, Hayes) is based on clinical need and only covers high-cost patients. Our K-means covers the whole population and reports OOP/income for each profile. This links the segmentation literature to the financial-burden literature (Banthin & Bernard). *Modestly novel. Clustering is descriptive, and results depend on K and scaling, so report stability checks.*
3. **The log-vs-dollar R² contrast as a teaching and methods point.** The same model can explain about 35% of variance in logs and about 0% in dollars. This shows concretely why risk-adjustment work reports dollar-scale R² and uses two-part models (Duan 1983; Manning & Mullahy 2001). *A useful demonstration, not a new finding.*
4. **Survey-design-aware ML benchmarking.** We compare CV-tuned KNN (and the pending XGBoost/ensembles) with OLS/logit using MEPS weights, and find no gain from flexibility. This matches Kan 2019, Rose 2016 and Buntin & Zaslavsky 2004 on a population-representative sample without claims data. *Novel within the scope of a class project. Few published papers use survey weights in ML cost prediction, but this was not checked systematically.*
5. **DML vs. OLS for demographic gaps (pending).** DML gives partially linear estimates of race/income gaps that adjust for health status flexibly. This links to the IOM definition (Cook/McGuire): which controls count as "need" determines the estimand. *Potentially the strongest contribution if framed as an IOM-style need adjustment with ML nuisance functions. Avoid causal language: race is not a manipulable treatment.*
6. **The Obermeyer link.** Our race gaps in spending at similar health status show directly why cost-labeled algorithms underserve Black and Hispanic adults. *Interpretive, not novel.*

---

## 4) References

- Banthin JS, Bernard DM. 2006. Changes in financial burden for health care: national estimates for adults age 18 to 64 years, 1996–2003. *JAMA* 296(22):2712–2719.
- Banthin JS, Cunningham P, Bernard DM. 2008. Financial burden of health care, 2001–2004. *Health Affairs* 27(1):188–195.
- Berk ML, Monheit AC. 2001. The concentration of health care expenditures, revisited. *Health Affairs* 20(2):9–18.
- Bertsimas D, Bjarnadóttir MV, Kane MA, et al. 2008. Algorithmic prediction of health-care costs. *Operations Research* 56(6):1382–1392.
- Blumenthal D, Chernof B, Fulmer T, Lumpkin J, Selberg J. 2016. Caring for high-need, high-cost patients: an urgent priority. *NEJM* 375:909–911.
- Buntin MB, Zaslavsky AM. 2004. Too much ado about two-part models and transformation? *J Health Econ* 23(3):525–542.
- Chernozhukov V, et al. 2018. Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal* 21(1):C1–C68.
- Cohen SB, Yu W. 2012. The concentration and persistence in the level of health expenditures over time: estimates for the U.S. population, 2008–2009. AHRQ MEPS Statistical Brief #354. [V~]
- Cook BL, McGuire TG, Zaslavsky AM. 2012. Measuring racial/ethnic disparities in health care: methods and practical issues. *Health Serv Res* 47(3 Pt 2):1232–1254.
- Cook BL, McGuire TG, Lock K, Zaslavsky AM. 2010. Comparing methods of racial and ethnic disparities measurement across different settings of mental health care. *Health Serv Res* 45(3):825–847. [V~]
- Dickman SL, Gaffney A, McGregor A, et al. 2022. Trends in health care use among Black and White persons in the US, 1963–2019. *JAMA Network Open* 5(4):e227403. [venue corrected from the prompt]
- Dieleman JL, Chen C, Crosby SW, et al. 2021. US health care spending by race and ethnicity, 2002–2016. *JAMA* 326(7):649–659.
- Duan N. 1983. Smearing estimate: a nonparametric retransformation method. *JASA* 78(383):605–610.
- Duan N, Manning WG, Morris CN, Newhouse JP. 1983. A comparison of alternative models for the demand for medical care. *J Bus Econ Stat* 1(2):115–126.
- Ellis RP, Martins B, Rose S. 2018. Risk adjustment for health plan payment. In McGuire TG, van Kleef RC (eds.), *Risk Adjustment, Risk Sharing and Premium Regulation in Health Insurance Markets*. Academic Press. [V~]
- Hadley J, Holahan J, Coughlin T, Miller D. 2008. Covering the uninsured in 2008. *Health Affairs* 27(5):w399–w415. [V~]
- Hayes SL, Salzberg CA, McCarthy D, et al. 2016. High-need, high-cost patients: who are they and how do they use health care? Commonwealth Fund Issue Brief. [V~]
- Joynt Figueroa J, Zhou X, Jha AK, Orav EJ. 2017. Segmenting high-cost Medicare patients into potentially actionable cohorts. *Healthcare* 5(1–2):62–67. [V~]
- Kan HJ, Kharrazi H, Chang HY, et al. 2019. Exploring the use of machine learning for risk adjustment: a comparison of standard and penalized linear regression models in predicting health care costs in older adults. *PLoS ONE* 14(3):e0213258. [V~]
- Kreif N, DiazOrdaz K. 2019. Machine learning in policy evaluation: new tools for causal inference. *Oxford Research Encyclopedia of Economics and Finance*. [V~]
- Mahajan S, Caraballo C, Lu Y, et al. 2021. Trends in differences in health status and health care access and affordability by race and ethnicity in the United States, 1999–2018. *JAMA* 326(7):637–648.
- Manning WG, Mullahy J. 2001. Estimating log models: to transform or not to transform? *J Health Econ* 20(4):461–494.
- Manning WG, Newhouse JP, Duan N, et al. 1987. Health insurance and the demand for medical care: evidence from a randomized experiment. *AER* 77(3):251–277.
- McGuire TG, Alegría M, Cook BL, Wells KB, Zaslavsky AM. 2006. Implementing the Institute of Medicine definition of disparities. *Health Serv Res* 41(5):1979–2005.
- Morid MA, Kawamoto K, Ault T, Dorius J, Abdelrahman S. 2017. Supervised learning methods for predicting healthcare costs: systematic literature review and empirical evaluation. *AMIA Annu Symp Proc* 2017:1312–1321.
- Obermeyer Z, Powers B, Vogeli C, Mullainathan S. 2019. Dissecting racial bias in an algorithm used to manage the health of populations. *Science* 366(6464):447–453.
- Pope GC, Kautter J, Ellis RP, et al. 2004. Risk adjustment of Medicare capitation payments using the CMS-HCC model. *Health Care Financ Rev* 25(4):119–141.
- Rose S. 2016. A machine learning framework for plan payment risk adjustment. *Health Serv Res* 51(6):2358–2374. **Note:** the *Biometrics* paper in the prompt may be Rose S. 2018, "Robust machine learning variable importance analyses of medical conditions for health care spending," *Health Serv Res*, or a *Biometrics* paper with a similar title. **Venue unverified. Check before citing.** [U]
- Shrestha A, Bergquist S, Montz E, Rose S. 2018. Mental health risk adjustment with clinical categories and machine learning. *Health Serv Res* 53(S1):3189–3206. [V~]
- van de Ven WPMM, Ellis RP. 2000. Risk adjustment in competitive health plan markets. In Culyer & Newhouse (eds.), *Handbook of Health Economics* Vol. 1A, ch. 14.
- Yang C, Delcher C, Shenkman E, Ranka S. 2018. Machine learning approaches for predicting high cost high need patient expenditures in health care. *BioMed Eng OnLine* 17(Suppl 1):131.

---
## Citation verification (web-checked 2026-11-13)
- **Dieleman JL et al. (2021). "US Health Care Spending by Race and Ethnicity, 2002–2016." *JAMA* 326(7):649–659.** Verified. In 2016, non-Hispanic White people made up 61% of the population but received 72% of spending. Gaps persisted after adjusting for age and health conditions. Use this paper as the multi-group race benchmark. https://jamanetwork.com/journals/jama/fullarticle/2783068
- **Dickman SL et al. (2022). "Trends in Health Care Use Among Black and White Persons in the US, 1963–2019." *JAMA Network Open*.** Verified that it is in JAMA Netw Open, not JAMA. It covers Black vs. White only: per-capita use by White people exceeded use by Black people in every year.
- **Rose S (2016). "A Machine Learning Framework for Plan Payment Risk Adjustment." *Health Services Research* 51(6).** Verified. Uses MarketScan 2011–2012 data and finds that smaller formulas selected by ML keep most of the efficiency of the full formula.
- **AHRQ MEPS Statistical Briefs (2018 data, released Jan 2021).** Verified. The top 5% accounted for 48.3% of spending, and the bottom 50% for 3.2%. The top 1% accounted for 21.9% (2016 data, Brief #521). Earlier years (2009–2015) put the top 5% at about 50–51%. https://meps.ahrq.gov/data_files/publications/st533/stat533.shtml

---
## Added sources: focused question on OOP spending by race × income × insurance (from user-supplied summary, 2026-11-13)
- **AHRQ MEPS analysis of 2022 expenditures** (meps.ahrq.gov): adults aged 65+ were 18.1% of the population but 40.5% of the top 5% of spenders. 75.1% of adults in the top 5% had two or more priority conditions. Non-Hispanic White people are overrepresented above the median and Hispanic people underrepresented. **Our replication:** 65+ adults are 22.6% of *adults* (AHRQ's base includes children) and 40.8% of the top 5%. 62.1% of the top 5% have two or more of *our six* conditions, a narrower list than AHRQ's priority conditions.
- **Health-care financing burden by income** (MEPS linked with other national sources; PubMed): higher-income households pay more dollars, but lower-income households pay a larger share of income. That measure includes premiums and taxes.
- **CMS national health expenditure by sex (2020):** working-age women spent about 20% more per person than men. This is national accounts data, not MEPS. **Ours:** women $8,720 vs. men $6,759 across all adults.
- **"Health Care Expenditures for Black and White US Adults Living Under Similar Conditions" (2023, MEPS, cross-sectional; PubMed):** Black–White spending differences shrank or disappeared under similar neighborhood racial and economic conditions and equitable access. A race coefficient alone does not identify mechanisms. *Author and journal not verified here.*

### How the focused results relate
Out-of-pocket dollars and burden tell different stories, as the financing literature predicts. Within every income group, Black and Hispanic adults have *lower* OOP dollars *and* a lower probability of being in a high-burden family than White adults. In the <200% FPL group, 25.7% of White adults are in high-burden families, against 11.4% of Black and 10.8% of Hispanic adults. Adjusting for income *widens* the burden gap (Black: −3.4 to −5.3 pp; Hispanic: −4.0 to −5.9 pp). The gaps persist after adding insurance and health status. Following the access literature, lower OOP may reflect less use of care (unmet need) rather than less financial strain. These are descriptive associations, not causal effects.
