

## Contributors

- Vihaan Manchanda — Linear regression (`linear_regression_vihaan_manchanda.ipynb`)
- Gaurav Law — Logistic regression (`gaurav_law_logistic.ipynb`)
- Anvita Suresh — Generalized Additive Model (`anvita_suresh_gam.ipynb`)

## Dataset

We used the Telco Customer Churn dataset. It has 7,043 customers and 21 columns. Each row is
one customer, with their account info (how long they've been with the company, contract type,
payment method), which services they pay for (phone, internet, streaming, and so on), what
they're charged (monthly and total), and a `Churn` column saying whether they left.

`TotalCharges` came in as text with a few blank cells for brand new customers, so we cleaned
that up first and dropped the blanks, leaving about 7,032 rows. Around 27% of customers
churned, so there are a lot more "stayed" customers than "left" ones. That imbalance comes
back later, because it makes the models worse at catching the people who actually leave.

The goal was two things at once: predict who churns, and be able to explain why the model
thinks so, since this assignment is really about interpretability.

## Assumption Checks

| Model | Key Assumptions Checked | Evidence | Concern |
|---|---|---|---|
| Linear regression | Straight-line relationships, equal spread of errors, features that don't overlap too much | Residual plot lands on two lines instead of a random cloud; churn rate vs tenure falls in a roughly straight line; tenure and TotalCharges correlate at 0.83 | Churn is 0/1, so the straight-line and equal-spread assumptions don't really hold. Some predictions even come out below 0, which can't be a probability |
| Logistic regression | Binary outcome, independent rows, straight-line relationship on the log-odds, features that don't overlap, enough churn cases | Confirmed Churn is 0/1; Box-Tidwell check on tenure and MonthlyCharges; VIF table; about 48 churn cases per feature (rule of thumb is 10–15+) | The one-hot encoding kept every category, so the features overlap almost perfectly and the VIF values blow up. The individual coefficients are hard to trust until that encoding is fixed |
| GAM | Feature effects are allowed to bend (splines are worth using), features that don't overlap, effects that add up independently | Churn vs tenure clearly curves (drops fast, then flattens), so a straight line would miss it; tenure and TotalCharges correlate at 0.83; the smooth terms come out significant | It assumes each feature acts on its own and doesn't model interactions. The p-values from the library are also known to be unreliable, and a couple of the curves look wiggly |

## Model Comparison

| Model | Performance Evidence | Interpretability Strength | Interpretability Weakness |
|---|---|---|---|
| Linear regression | R² = 0.24, RMSE = 0.38, accuracy 78% when predictions are rounded at 0.5 | One number per feature, so it's easy to read which way each feature pushes churn | The output isn't a real probability (it can go below 0 or above 1), and the raw coefficients are on different scales, so you can't compare their sizes directly |
| Logistic regression | Accuracy 82%, pseudo-R² = 0.22; on the churners it gets precision 0.68, recall 0.60, F1 0.64 | Gives odds ratios you can say in plain words (month-to-month about 1.36x the odds, fiber about 1.42x), and it predicts proper probabilities between 0 and 1 | Each feature is assumed to act in a straight, additive way on the log-odds, so it misses curves and interactions on its own. The overlapping features also throw off the odds ratios |
| GAM | Accuracy 80%, AUC 0.84; recall on the churners is only 0.54, so it still misses about half of them | The curves show the shape of each effect, like tenure dropping steeply and then leveling off, which a single coefficient can't show | You have to add interactions by hand, the p-values aren't trustworthy, and the wavier curves are harder to explain to a non-technical audience |

## Recommendation

Recommended model: Logistic regression.

Why this model: It had the best numbers overall, with the highest accuracy (82%) and the best
score at actually catching churners (recall 0.60, F1 0.64). It also gives probabilities and
odds ratios that are easy to hand to a business team and act on. Linear regression is the
wrong fit for a yes/no question, since its predictions leave the 0-to-1 range and its
assumptions don't hold. The GAM is interesting because it shows the curved effects, but it
didn't score any better and its p-values can't be trusted. So logistic is the best mix of
accuracy, clear explanations, and simplicity. The one thing to fix first is the encoding, so
the overlapping features stop distorting the odds ratios.

What the company can responsibly conclude: All three models point the same way, and that
agreement is the strongest thing going for these findings. Month-to-month contracts, fiber
optic internet, higher charges, and shorter tenure all line up with more churn, while longer
contracts and longer tenure line up with less. Contract type and tenure stand out as the
biggest levers.

What the company should not conclude yet: These are associations, not causes. We can't say
that moving someone to a two-year contract will make them stay, only that the two tend to go
together. The exact sizes of the effects aren't reliable yet either, because of the
overlapping features in the logistic model and the class imbalance. And since every model
still misses roughly 40–46% of the customers who actually leave, this isn't accurate enough to
base expensive, customer-by-customer retention offers on.

One next analysis we would run: Handle the class imbalance (class weights or resampling) and
refit, then check whether recall on the churners improves, after cleaning up the encoding so
the odds ratios settle down. After that, we'd test a tenure-by-contract interaction to see if
the effect the additive models can't capture actually matters.
