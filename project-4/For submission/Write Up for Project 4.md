A brief writeup in an executive summary, written for a non-technical audience.
   - Writeups should be at least 500-1000 words, defining any technical terms, explaining your approach, as well as any risks and limitations.

### Scraping 

I created a list of job titles base on my general knowledge regarding the data science industry. 

search_terms = ['data analyst','data engineer','data scientist','business intelligence','data architect',
                'machine learning','deep learning','Natural Language Processing','business analytics',
                'data science','BI','NLP','solution architecture','data mining','predictive analytics','database'
                ,'analytics']

Using this list, I manage to gathered 3234 job posts , but the job posting search was limited to this fixed list. 

The list could be not comprehensive enough to scrape all related jobs. 

After getting all the individual URL for the 3234 jobs, I went ahead to scrap 11 features from the job postings. 

data = {'URL':url_list,'company':company,'job_title':job_title,'sal_lower':sal_lower,
        'sal_upper':sal_upper,'sal_type':sal_type,'job_requirements':requirements,
       'skills':skills, 'emp_type':emp_type,'seniority':seniority,
       'industry':industry,'experience':experience}
       
I did not scrap the location and date posted. Based on my general knowledge working in Singapore, the salary should not deviate much based on the location of job, because Singapore is relatively small. As for the date posted, I believe it should not be used as a factor for predicting the salary, as the date could be any date and any amount of days, based on when the job poster decided to remove it. 

After scraping all the features, I ran a duplicate check and the number of job listings was reduced to 2016.


### Question 1  : 

##### Cleaning approach : 
I loaded the data that was scrapped, and started some basic cleanings. All apostrophes and square brackets was removed as they will not have any meaning for my models. 

##### Filtering non-related jobs approach: 
I decided to further filter the jobs that I have, by creating a list of data science skills keywords and matching them with all the job posting skills. 

["analytic" , "analysis", "machine", "data" ,"intelligence", "statistic" ,"solution", "image processing" , "mining" , "recognition", "predictive"]

The risk of using this list is the same as previous approach. The list might not be comprehensive enough to filter all the right jobs. 

After removing jobs that are potentially not related. I am left with 1223 jobs that contains atleast 1 data science related skills.

#### Standardising the employment type column 

Many of the job posts has irrational number of employment type. It seems like the job posters are attempting to hire a range of different positions in one job posting. 

I decided to reduce all job posting's employment type to the lowest denominator. 

Firstly, I came out with a ranking list for employment type. 

Assume the ranks of employment type using general knowledge and information from careersfuture.

From lowest to highest

1:Temporary , 2:Internship , 3:Flexi work ,4:Freelance , 5:Part Time ,6:Contract ,7:Full time,8:Permanent .

Using this list, I replace all the job employment type for individual job postings to the lowest denominator. 

By doing so, I risk losing some information about the job employment type. After standising the employment type, I decided that I should standardise everything to the lowest denominator to keep it consistent. 

#### Standardising the Seniority 


Many of the job posts has irrational number of seniority type. It seems like the job posters are attempting to hire a range of different seniority in one job posting. I applied the same approach for employment type column. 

Rank the 9 main groups accordingly. Follow mycareersfuture.sg

Lowest to highest

1:Fresh/Entry Level,2:Non-executive,3:Junior Executive,4:Executive,5:Senior Executive,6:Professional,7:Manager,8:Middle Management,9:Senior Management

#### Dummifying the industry column 

After exploring the industry column , I discovered that, there is 37 unique industry. Each job post can be a mixture of industry. 

I decided to dummify the industry information , as I believe there might be high correlation between industry and salary amount. 


#### Dropping company names

I decided to drop the company names as there was too many unique names. It will not give any meaningful insights. 

But in reality , there is a high chance that certain companies tend to pay a better renumeration package. ( For eg, Google ) 


#### Dummifying the skills column 

There are alot of unique skills, but I decided to dummify and keep all the skills, because i believe there is high correlation between skills and the salary amount. 

The risk for this particular approach is , the complexity will increased greatly, as there is a huge number of unique skills. 

#### Decided to take salary lower to build my label 

I decided to take salary lower and drop salary upper, as I have already standardised both employment type and seniority to the lowest denominator.

After drop salary upper, I decided to use median as the threshold to split salary into low / high labels that will be used as target for my predictive models. 


#### Applied NLP on job title and job requirement

I used countvectoriser with a setting of n_gram(2,3) to get a list of high appearance keywords with the length of 2-3 words. 

By using n_gram(2,3) , there is a risk of missing out meaningful single words. 

But I decided to stick to (2,3) as this option filters away alot of useless single words. 

#### Classification models : Logistic Regression and Random Forest 

I did a grid search for both models to get the optimal hyper parameters. 

The best model turn out to be logistic regression. 

===Logistic Regression===
Best Param: {'C': 0.1, 'penalty': 'l2'}
Best AUC Score from CV: 0.8485862719219104
prediction score: 0.8341836734693878
AUC Score(Test): 0.8341836734693878

coef	variable
-0.785418	seniority_exec
0.627207	seniority_midman
-0.563173	skill_microsoft office
0.560302	industry_Banking and Finance
-0.510113	seniority_fresh
0.404361	10 years
-0.401397	team player
-0.396816	data visualization
0.386510	good understanding
0.369259	seniority_manager


Executive tend to be in the lower range of the salary vs mid management/managers.

Microsoft office is a irrelevant skill to get higher paying salary.

The banking industry seems to be the best paying industry. 

Data visualisation skills seems to command lower salary. 



### Question 2  : 

Similar cleaning and data preparation was done for question 2 using the same dataset. 

The label was changed to Job title instead of salary category. 

I decided to do a classification problem for "Data scientist" vs "Other data jobs" 

My label for question 2 : 1 for data scientist , 0 for any other data jobs. 

The dataset was highly imbalance, with only 84(data scientist) vs 1138(others). 

Decided to go ahead with building models on the original dataset. 

Logistic regression performed better than Random Forest. 

===Logistic Regression===
Best Param: {'C': 1, 'penalty': 'l2'}
Best AUC Score from CV: 0.8905397277976723
Prediction score: 0.9387134502923976
AUC Score: 0.9387134502923976
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       342
           1       0.64      0.64      0.64        25

   micro avg       0.95      0.95      0.95       367
   macro avg       0.81      0.81      0.81       367
weighted avg       0.95      0.95      0.95       367

Predicted    0   1  All
Actual                 
0          333   9  342
1            9  16   25
All        342  25  367

The f1-Score for label 1 wasn't very good as the dataset imbalance. 

Decided to try up sampling using SMOTE to see if there will be any significant improvement. 

===Logistic Regression===
Best Param: {'C': 1, 'penalty': 'l2'}
Best AUC Score from CV: 0.9881309554849722
Prediction score: 0.9375438596491228
AUC Score: 0.9375438596491228
              precision    recall  f1-score   support

           0       0.99      0.93      0.96       342
           1       0.48      0.84      0.61        25

   micro avg       0.93      0.93      0.93       367
   macro avg       0.73      0.89      0.78       367
weighted avg       0.95      0.93      0.94       367

Predicted    0   1  All
Actual                 
0          319  23  342
1            4  21   25
All        323  44  367


The recall improved but the f1-score has dropped due to a low precision score. 


Decided to use the first logistic regression model with original dataset, as the scores are more balance for both recall and precision. 

Top 10 coefficients to predict data scientist job : 

coef	variable
1.881373	machine learning
1.675398	data science
1.078439	product development
0.949163	experience data
0.924179	industry_Logistics / Supply Chain
0.901366	work experience
0.800171	seniority_fresh
0.731601	deep learning
0.676066	related field
0.620938	relevant experience


Machine learning and keyword "data science" in the job description has the highest weightage for predicting "data scientist" jobs . 

Seems like data scientist are normally involved in product development, based on the coefficient. 

Data scientist are most employed in the logistics / supply chain industry. ( A little bit counter-intuitive ) 

Deep learning is a key skill for data scientist. 

Relevant experience and related field is a important feature for a data scientist. 
