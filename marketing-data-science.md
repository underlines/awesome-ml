
# awesome-datascience

Collection of all things "Data Science" leaning towards Marketing & Advertisement (Digital Marketers, Agencies, Web Designers and Web Analysts)

# Table of Contents
 - [Software](#software)
   - [Data Architecture](#data-architecture)
     - [Classic ETL](#classic-etl)
     - [Cloud ETL](#cloud-etl)
     - [Metadata](#metadata)
     - [Datastores / Databases](#datastores--databases)
     - [Customer Data Platforms](#customer-data-platforms)
     - [Job Scheduling / Workflows / Orchestration / Transformation](#job-scheduling--workflows--orchestration--transformation)
   - [Web Analytics / App Analytics](#web-analytics--app-analytics)
   - [Machine Learning IDEs](#machine-learning-ides)
   - [BI / Data Visualization / Reporting](#bi--data-visualization--reporting)
   - [SaaS Marketing Reporting Solutions](#SaaS-Marketing-Reporting-Solutions)
   - [Deployment Environments](#deployment-environments)
 - [Libraries](#libraries)
   - [Python](#python)
     - [Visualization](#visualization)
     - [Machine Learning](#machine-learning)
     - [Digital Marketing](#digital-marketing)
 - [Learning](#learning)
   - [Courses](#courses)
     - [Free Courses](#free-courses)
     - [Paid Courses](#paid-courses)
     - [Free interactive learning websites](#free-interactive-learning-websites)
     - [Paid interactive learning websites](#paid-interactive-learning-websites)
   - [Books](#books)
     - [Statistics & Mathematics](#statistics--mathematics)
     - [Data Visualization](#data-visualization)
     - [Data Science in general](#data-science-in-general)
     - [Machine Learning](#machine-learning)
     - [Data Architecture](#data-architecture)
   - [Other Resources](#other-resources)

# Software
## Data Architecture
- [Snowflake | Cloud Data Platform](https://www.snowflake.com/)
- [Spark | Distributed data processing framework for ML, analytics and more](https://spark.apache.org/)
- [Databricks | Cloud platform for data engineering and data science by the makers of Spark (Community Edition available)](https://databricks.com/)
- [Arc | Open-source Databricks alternative](https://arc.tripl.ai/)
- [Supabase | open source Firebase alternative](https://github.com/supabase/supabase)
- [UKV | Replacing MongoDB, Neo4J, and Elastic with 1 transactional database. Features: zero-copy semantics, swappable backends, bindings for C, C++, Python, Java, GoLang](https://github.com/unum-cloud/ukv)


### Classic ETL
- Talend Open Studio
- Tibco Jaspersoft ETL
- [Halzelcast Jet-start.sh - Distributed Streaming](https://jet-start.sh/)
- [Mode | Interactive data science meets modern BI for fast, exploratory analysis company-wide](https://mode.com/)
- [DBT | Open-source tool to organize, cleanse, denormalize, filter, rename, and pre-aggregate raw data in warehouse for analysis.](https://github.com/fishtown-analytics/dbt)
- [AirByte | Open-source ELT solution for simple data integration, owned by you](https://github.com/airbytehq/airbyte)
- [PRQL | a modern language for transforming data — a simple, powerful, pipelined SQL replacement](https://prql-lang.org/)
- [YoBulk | Opensource CSV importer powered by GPT3, flatfile alternative](https://github.com/yobulkdev/yobulkdev)
- 

### Cloud ETL
- [Singer | Open-source composable data extraction for many sources and destinations](https://www.singer.io/)
- [Fivetran](https://fivetran.com/)
- [Stitchdata](https://www.stitchdata.com/)
- [Panoply](https://panoply.io/)
- [Electrik.ai | Extract Raw hit level Google Anayltics data](https://electrik.ai/)
- [OWOX | ETL specialized for Digital Marketing purposes](https://www.owox.com/)
- [Scitylana | Extract Raw hit level Google Analytics data](https://www.scitylana.com/)
- [Matillion | ETL & Transformation](https://www.matillion.com/)
- [funnel.io | Marketing Data Extraction](https://funnel.io/)
- [Adverity | Marketing Data Integration, Reporting and Analytics](https://www.adverity.com/)
- [Singer | Open Source Data Integration](https://github.com/singer-io)

### Metadata
- [Metacat | Open-source Metadata management for Hive, RDS, Teradata, Redshift, S3 and Cassandra](https://github.com/Netflix/metacat)
- [Amundsen Frontend Service | Open-source Metadata indexing for tables, dashboards, streams, etc. with page-rank style search](https://github.com/amundsen-io/amundsenfrontendlibrary)

### Datastores / Databases
- [TimescaleDB | open-source database for scalable SQL time-series based on PostgreSQL](https://github.com/timescale/timescaledb)
- [EventNative | open source, high-performance, event collection service](https://github.com/ksensehq/eventnative)
- [Apache Ignite | open-source in-memory distributed database, caching, and processing platform for transactional, analytical, and streaming workloads ](https://ignite.apache.org/)
- [Redis | open source in-memory advanced key-value store used as a database, cache and message broker](https://redis.io/)
- [Apache Kafka | Open-source distributed event streaming platform for high-performance data pipelines, streaming analytics, data integration](https://kafka.apache.org/)
- [InfluxDB | open-source time series database](https://github.com/influxdata/influxdb)
- [ClickHouse | open-source column oriented OLAP DBMS for realtime querying in SQL](https://github.com/ClickHouse/ClickHouse)
- [RQLite | open-source Lightweight, distributed SQLite database handling leader elections, tolerates failures of machines, including leader available for Linux, macOS, and Microsoft Windows](https://github.com/rqlite/rqlite)
- [DuckDB | open-source fast in-process SQL OLAP database](https://github.com/duckdb/duckdb)



### Customer Data Platforms
- [Segment | Marketing CDP](https://segment.com/)
- [Ascent360 | Full-stack CEP/CDP](https://www.ascent360.com/)
- [SAP Emarsys | Full-stack CDP](https://emarsys.com/)
- [Firsthive | Marketing CDP to take control of first-party data from online and offline, and enable personalized campaigns](https://firsthive.com/)
- [Dynamics 365 Customer Insights | Full-stack CDP ](https://dynamics.microsoft.com/en-us/ai/customer-insights/)
- [Piwik.pro | Marketing CDP](https://piwik.pro/customer-data-platform/)
- [Salesforce Customer Interaction | Full-stack CDP](https://www.salesforce.com/products/marketing-cloud/customer-interaction/)
- [Tealium Audience Stream | Marketing CDP](https://tealium.com/products/audiencestream/)
- [Treasuredata | Full-stack CDP](https://www.treasuredata.com/)
- [Blueshift | Full-stack & Marketing CDP](https://blueshift.com/)
- [Exponea | Marketing CDP](https://exponea.com/)
- [Rudderstack | Paid Customer Data Pipeline for Event streaming, Warehouse sync and ETL, open-source community edition available](https://rudderstack.com/)

### Job Scheduling / Workflows / Orchestration / Transformation
- [Apache Airflow | Workflow scheduler using Directed Acyclical Graphs in Python](https://airflow.apache.org/)
- [Apache Oozie | Workflow scheduler to manage Hadoop jobs as Directed Acyclical Graphs in Java and XML](oozie.apache.org)
- [Luigi | open-source pipeline and batch job management](https://github.com/spotify/luigi)
- [ActivePieces | Open Source Zapier alternative for business automation](https://github.com/activepieces/activepieces)
- [Automatisch | Open SOurce Zapier alternative for business automation](https://github.com/automatisch/automatisch)

## Web Analytics / App Analytics
- [Mixpanel | self-serve product analytics to help you convert, engage, and retain more users](https://mixpanel.com/home/)
- [Amplitude | Digital product and user analytics platform](https://amplitude.com/)
- [Snowplow | Open source Web, mobile and event analytics for AWS and GCP](https://github.com/snowplow/snowplow)
- [RRWeb | Open source web session recorder & player for user behaviour analysis](https://github.com/rrweb-io/rrweb)
- [Plausible Analytics | Open-source lightweight, privacy respecting cookie-less Google Analytics alternative](https://github.com/plausible/analytics)
- [Simple Analytics | Paid cookie-less, privacy respecting Google Analytics alternative](https://simpleanalytics.com/)
- [Fathom Lite | Open-source self hosted Google Analytics alternative](https://github.com/usefathom/fathom)
- [Shynet | Open-source cookie-less Google Analytics alternative](https://github.com/milesmcc/shynet)
- [Friendly Analytics | Paid privacy respecting Matomo/Piwik fork](https://friendly.is/en/analytics)
- [Google Analytics 4 | Cookie & Cookie-less App & Web Analytics platform with flexible event model and tight Google Marketing Cloud integration](https://searchengineland.com/google-analytics-4-adds-new-integrations-with-ads-ai-powered-insights-and-predictions-342048)
- [Cloudflare Web Analytics | Free cookie-less privacy respecting Google Analytics alternative](https://www.cloudflare.com/web-analytics/)
- [Panelbear | Paid Cookie-less Google Analytics alternative](https://panelbear.com/)
- [Open Web Analytics | Open-source Google Analytics alternative](https://github.com/Open-Web-Analytics/Open-Web-Analytics)
- [GoAccess | Open-source realtime web-log analytics for terminal and web](https://goaccess.io/)
- [Matomo | (Formerly Piwik) Open-source Google Analytics alternative](https://matomo.org/)
- [Analytics | Lightweight Open-source abstraction layer for web analytics and marketing tracking](https://github.com/DavidWells/analytics)
- [Umami | Open-source, light weight Google Analytics alternative](https://github.com/mikecao/umami)
- [Google Analytics Beacon | A proxy for Universal Analytics allows to track via image pixel, when no javascript is allowed](https://github.com/igrigorik/ga-beacon)
- [Getinsights | Privacy-focused, cookie free analytics, free for up to 5k events/month.](https://getinsights.io/)
- [Keen.io | Managed Event Streaming Platform, built on Kafka, Storm, and Cassandra](https://keen.io/)
- [Quantcast Analytics | Audience Analytics with 3rd party data exchange](https://www.quantcast.com/products/measure-audience-insights/)
- [Engauge | Open-source single binary app and web analytics with minimal requirements](https://github.com/EngaugeAI/engauge)
- [Counter.dev | Basic, open source non-intrusive web analytics](https://github.com/ihucos/counter.dev)


## Machine Learning IDEs
- [KNIME Analytics | open source software for creating visual data science workflows](https://www.knime.com/knime-analytics-platform)
- [Jupyter Hub / Lab / Notebooks | Open source single & multiuser IDE for data science and machine learning supporting Python and more](https://jupyter.org/)
- [Knowage | Open source BI and Analytics suite](https://www.knowage-suite.com/site/)
- [Microsoft R Open | open source platform for statistical analysis and data science](https://mran.microsoft.com/open)
- [Orange3 | Open source machine learning and data visualization and visual data analysis workflows](https://orange.biolab.si/)
- [RanalyticFlow | Open source data analysis software built on R, for interactive data analysis with or without R programming](https://r.analyticflow.com/en/)
- [Rapidminer Studio | data science platform that unites data prep, machine learning & predictive model deployment](https://rapidminer.com/)
- [RStudio | Desktop and cloud based single or collaborative IDE for R](https://rstudio.com/)

## BI / Data Visualization / Reporting
- [Tibco Spotfire Analytics | AI-powered, search-driven experience with built-in data wrangling and advanced analytics](https://www.tibco.com/products/tibco-spotfire)
- [Power BI](https://powerbi.microsoft.com/en-us/)
- [Tableau](https://www.tableau.com/)
- [Pentaho Business Analytics](https://www.hitachivantara.com/en-us/products/data-management-analytics/pentaho-platform/pentaho-business-analytics.html)
- [Panoply | Full service Analytics Stack](https://panoply.io/platform/)
- [QlikSense | Assisted BI Analytics suite](https://www.qlik.com/us/products/qlik-sense)
- [Klipfolio | metrics, meaningful dashboards, and actionable reports](https://www.klipfolio.com/)
- [Geckoboard | Self service dashboards for various Data Sources](https://www.geckoboard.com/)
- [Google Data Studio | Free dashboard and reporting service focused on marketing data sources](https://datastudio.google.com/)
- [Sisense | BI Analytics suite](https://www.sisense.com/)
- [Looker | BI Analytics suite](https://looker.com/)
- [Canopy.cloud | Reporting and Analytics for Investors and Wealth managers](https://canopy.cloud/)
- [Chartio | Dashboard and Reporting platform](https://chartio.com/)
- [Cyfe](https://www.cyfe.com/)
- [Metabase | Open-source data query and visualization for non-tech people](https://github.com/metabase/metabase)
- [Redash | Open-source data query and visualization for non-tech people](https://github.com/getredash/redash)
- [Looker | Google acquired data visualization](https://looker.com)
- [Apache Grafana | Open-source data visualization and monitoring](https://github.com/grafana/grafana)
- [Apache Superset | Open-source enterprise grade web based BI solution](https://github.com/apache/superset)


## SaaS Marketing Reporting Solutions
- [AgencyAnalytics](https://agencyanalytics.com/)
- [WhatGraph](https://whatagraph.com/)
- [Reportz.io](https://reportz.io/)
- [Swydo](https://www.swydo.com/)
- [Megalytic](https://www.megalytic.com/)
- [Octoboard](https://www.octoboard.com/)
- [ReportingNinja](http://www.reportingninja.com/)
- [SuperDash](https://www.superdash.com/)
- [ReportGarden](https://reportgarden.com/)
- [DashThis](https://dashthis.com/)


## Deployment Environments
- [dstack - an open-source framework for building data science applications using Python and R](https://github.com/dstackai/dstack)
- [Analytics Zoo | Unified Data Analytics and AI stack with TensorFlow, Keras and Pytorch and seamless deployment](https://github.com/intel-analytics/analytics-zoo)

# Libraries
## Python
### Visualization
- [AutoViz](https://github.com/AutoViML/AutoViz), Automatically Visualize any dataset, any size with a single line of code.
- [PyGWalker | TUrn pandas dataframe into Tableau-style UI for Data Analysis](https://github.com?Kanaries/pygwalker)

### Machine Learning
- [MLBox | powerful Automated Machine Learning python library](https://github.com/AxeldeRomblay/MLBox)
- [PyCaret | open source low-code machine learning library in Python](https://github.com/pycaret/pycaret)
- [PyTorch Lightning | open-source Python library as a high-level interface for PyTorch](https://github.com/PyTorchLightning/pytorch-lightning)
- [igel | machine learning tool to train/fit, test and use models without writing code](https://github.com/nidhaloff/igel)
- [Ludwig | toolbox to train and evaluate deep learning models without writing code built on TensorFlow](https://github.com/uber/ludwig)
- [fast.ai | training fast and accurate neural nets using modern best practices](https://github.com/fastai/fastai)
- [Apache Arrow | Open-source cross-language development platform for in-memory analytics](https://arrow.apache.org/)
- [auto-sklearn](https://github.com/automl/auto-sklearn) | Automated Machine Learning for Python

### Digital Marketing
- [Google Ads Performance Pipeline | Data integration pipeline from GAds to PostgreSQL](https://github.com/mara/google-ads-performance-pipeline)
- [Text-Mining-Search-Query | segements search terms to search words and summarizes performance metrics](https://github.com/RachelPengmkt/Text-Mining-Search-Query)
- [CRMint | reliable data integration and processing for advertisers](https://github.com/google/crmint)
- [Official Google Ads Python library | client library for Google Ads API](https://github.com/googleads/google-ads-python)
- [Official Google Ads Python library | SOAP Ads APIs for AdWords and DoubleClick for Publishers](https://github.com/googleads/googleads-python-lib)
- [GAQL CLI| Running GoogleAds queries](https://github.com/getyourguide/gaql-cli)
- [advertools | SEM, SEO, Social productivity & analysis tools to scale your online marketing](https://github.com/eliasdabbas/advertools)
- [pyaw-reporting | AdWords API large scale reporting tool written in Python](https://pypi.org/project/pyaw-reporting/)

### Search
- [Typesense | fast, small, typo-tolerant search engine, when Elasticsearch is too big](https://github.com/typesense/typesense)
- [Elasticsearch | distributed RESTful search engine built for the cloud](https://github.com/elastic/elasticsearch)
- [Lucene Solr | enterprise search platform written in Java and using Apache Lucene](https://github.com/apache/lucene-solr)

### Query Engine
- [Trustfall | Query almost any data source as if it's SQL](https://github.com/obi1kenobi/trustfall)


# Learning
## Courses
### Free Courses
- Andrew Ng, Coursera, 54h [Machine Learning](https://www.coursera.org/learn/machine-learning)
- Herbert Lee, Coursera, 10h [Bayesian Statistics: From Concept to Data Analysis](https://www.coursera.org/learn/bayesian-statistics)
- Jeff Leek, Coursera/John Hopkins University, 300h [Data Science Specialization](https://www.coursera.org/specializations/jhu-data-science)
- CS109 Data Science, Harward [CS109](http://cs109.github.io/2015/pages/videos.html)
- Rafael Irizarry, Michael Love, Hardward/edX, 16h [Statistics and R](https://www.edx.org/course/statistics-and-r)
- Dave Holtz, Cheng-Han Lee, Udacity [Intro to Data Science](https://www.udacity.com/course/intro-to-data-science--ud359)
- Christopher Brooks, University of Michigan, 120h [Applied Data Science with Python Specialization](https://www.coursera.org/specializations/data-science-python)
- Saeed Aghabozorgi, IBM, 20h [Machine Learning with Python](https://www.coursera.org/learn/machine-learning-with-python)

### Paid Courses
- Jose Portilla, Udemy, 25h [Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/)
- https://www.udemy.com/course/datascience

### Free interactive learning websites
- Learnpython [Learnpython.org](https://www.learnpython.org/)
- W3 Schools [SQL](https://www.w3schools.com/sql/)

### Paid interactive learning websites
- [Qwiklabs | Learn on cloud environments and instances of aws, google cloud](https://www.qwiklabs.com/)

## Books
### Statistics & Mathematics
- [Bayesian Methods for Hackers](https://www.amazon.com/Bayesian-Methods-Hackers-Probabilistic-Addison-Wesley/dp/0133902838)
- [Naked Statistics: Stripping the Dread from the Data](https://www.amazon.com/Naked-Statistics-Stripping-Dread-Data-ebook/dp/B007Q6XLF2)
- [How to lie with statistics](https://www.amazon.com/How-Lie-Statistics-Darrell-Huff-ebook-dp-B00351DSX2/dp/B00351DSX2)
- [An Introduction to Statistical Learning: with Applications in R](https://www.amazon.com/Introduction-Statistical-Learning-Applications-Statistics/dp/1461471370)
- [Naked Statistics: Stripping the Dread from the Data](https://www.amazon.com/Naked-Statistics-Stripping-Dread-Data/dp/039334777X/)

### Data Visualization
- [Effective Data Visualization: The Right Chart for the Right Data](https://www.amazon.com/Effective-Data-Visualization-Right-Chart/dp/1544350880)
- [Data-Driven Storytelling (AK Peters Visualization Series](https://www.amazon.com/Data-Driven-Storytelling-AK-Peters-Visualization/dp/1138197106)
- [Making Data Visual: A Practical Guide to Using Visualization for Insights](https://www.amazon.com/Making-Data-Visual-Practical-Visualization/dp/1491928468)
- [Data Visualisation: A Handbook for Data Driven Design](https://www.amazon.com/Data-Visualisation-Handbook-Driven-Design/dp/1473912148)
- [Storytelling with Data: A Data Visualization Guide for Business Professionals](https://www.amazon.com/Storytelling-Data-Visualization-Business-Professionals/dp/1119002257)
- [Fundamentals of Data Visualization: A Primer on Making Informative and Compelling Figures](https://www.amazon.com/Fundamentals-Data-Visualization-Informative-Compelling/dp/1492031089)
- [The Art of Data Science](https://www.amazon.com/Art-Data-Science-Roger-Peng/dp/1365061469)
- [Python: Data Analytics and Visualization](https://www.amazon.com/Python-Analytics-Visualization-Phuong-Vo-T-H/dp/1788290097)
- [Dash for Python | Build web analytics application in Python in a few hours](https://dash-docs.herokuapp.com/introduction)

### Data Science in general
- [Python Data Science Handbook](https://www.amazon.com/Python-Data-Science-Handbook-Essential/dp/1491912057)
- [Big Data Science & Analytics: A Hands-On Approach](https://www.amazon.com/Big-Data-Science-Analytics-Hands/dp/0996025537)
- [Data Science from Scratch: First Principles with Python](https://www.amazon.com/Data-Science-Scratch-Principles-Python/dp/1492041130)
- [Everybody Lies: Big Data, New Data, and What the Internet Can Tell Us About Who We Really Are](https://www.amazon.com/Everybody-Lies-Internet-About-Really-ebook/dp/B01AFXZ2F4)

### Machine Learning
- [Deep Learning (Adaptive Computation and Machine Learning series)](https://www.amazon.com/Deep-Learning-NONE-Ian-Goodfellow-ebook/dp/B01MRVFGX4)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646/)

### Data Architecture
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.amazon.com/Designing-Data-Intensive-Applications-Reliable-Maintainable/dp/1449373321)

# Other Resources
- https://github.com/onurakpolat/awesome-analytics
- https://github.com/onurakpolat/awesome-bigdata
- https://github.com/academic/awesome-datascience
- https://github.com/fasouto/awesome-dataviz
- https://github.com/thenaturalist/awesome-business-intelligence
