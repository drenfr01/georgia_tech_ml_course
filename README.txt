My code is located at https://github.com/drenfr01/georgia_tech_ml_course under the folder project_1.

There are 2 main folders in my repository, a data directory which contains my 2 datasets "housing" and "DontGetKicked"
and a src file with all of my code.

My code in the src directory is structured into a main file along with 3 separate folders for helper_code, code
relevant for the housing problem, and code relevant to the automotive problem (i.e. lemons).

The code is modular, so the main.py file instantiates a Housing & Lemons object respectively. In the Housing & Lemons
folders you'll find a main file named either run_housing or run_lemons. These files use helper_files in the helper_files
folder to read in the datasets and define which features will be used.

Within housing & lemons you'll see one file per algorithm implemented. In the case of the neural networks you'll
also see 2 .ipynb notebooks. This is because the neural networks ran much faster on GPUs so I can the notebooks
on Google Colab.

In run_housing and run_lemons there is a function responsible for setting up and calling each algorithm. So e.g.
you can call self.run_svm or self.run_knn.

Each algorithm function in run_housing or run_lemons can call any of the functions in that respective algorithm file,
e.g. my_svm.py. These functions are named logically and can do everything from running cross validation to
creating a learning curve.

It's worth noting that to graph functions I wrote the output to csv files and used Excel to create the graphs

