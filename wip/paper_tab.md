\begin{table}
\caption{Resultats}
\label{tab:results}
\makebox[\textwidth][c]{
\begin{tabular}{lccccccc}
\toprule
Tâches & ESN & GRU & LSTM & MAMBA & TRANS. (DO) & TRANS. (ED) & XLSTM \\
\midrule
adding problem-SM-1k                   & 0.47±0.09      & \textbf{0.08±0.05}  & 0.50±0.31      & 0.63±0.21      & 0.56±0.24            & 0.69±0.06           & 0.32±0.37      \\
adding problem-SM-10k                  & \textbf{0.02±0.03}  & 0.13±0.13      & 0.36±0.33      & 0.63±0.20      & 0.15±0.22            & 0.42±0.27           & 0.07±0.06      \\
adding problem-SM-100k                 & nan              & 0.16±0.10      & 0.55±0.29      & 0.61±0.23      & \textbf{0.04±0.03}        & 0.22±0.32           & 0.06±0.07      \\
adding problem-MD-1k                   & 0.85±0.02      & \textbf{0.01±0.01}  & 0.02±0.01      & 0.88±0.01      & 0.70±0.29            & 0.88±0.01           & 0.44±0.45      \\
adding problem-MD-10k                  & 0.71±0.02      & 0.01±0.00      & 0.01±0.01      & 0.23±0.37      & 0.01±0.01            & 0.54±0.44           & \textbf{0.00±0.00}  \\
adding problem-MD-100k                 & nan              & 0.01±0.00      & 0.02±0.01      & 0.23±0.38      & 0.01±0.01            & 0.20±0.36           & \textbf{0.00±0.00}  \\
associative rec.-SM-1k               & \textbf{0.57±0.07}  & 0.66±0.03      & 0.66±0.04      & 0.65±0.04      & 0.64±0.04            & 0.65±0.05           & 0.64±0.05      \\
associative rec.-SM-10k              & \textbf{0.52±0.04}  & 0.58±0.11      & 0.65±0.04      & 0.66±0.04      & 0.59±0.12            & 0.61±0.07           & 0.56±0.05      \\
associative rec.-SM-100k             & nan              & 0.57±0.11      & 0.65±0.07      & 0.66±0.04      & 0.55±0.09            & 0.56±0.10           & \textbf{0.52±0.07}  \\
associative rec.-MD-1k               & \textbf{0.85±0.01}  & 0.93±0.01      & 0.93±0.01      & 0.93±0.01      & 0.92±0.01            & 0.93±0.01           & 0.92±0.01      \\
associative rec.-MD-10k              & \textbf{0.78±0.01}  & 0.93±0.01      & 0.93±0.00      & 0.93±0.01      & 0.86±0.03            & 0.90±0.03           & 0.91±0.01      \\
associative rec.-MD-100k             & nan              & 0.90±0.01      & 0.93±0.01      & 0.93±0.01      & \textbf{0.84±0.07}        & 0.90±0.06           & 0.88±0.01      \\
bracket matching-SM-1k                 & \textbf{0.25±0.11}  & 0.30±0.09      & 0.33±0.06      & 0.39±0.05      & 0.45±0.08            & 0.48±0.06           & 0.38±0.06      \\
bracket matching-SM-10k                & \textbf{0.20±0.06}  & 0.35±0.03      & 0.36±0.06      & 0.38±0.04      & 0.46±0.05            & 0.46±0.06           & 0.28±0.10      \\
bracket matching-SM-100k               & nan              & 0.39±0.11      & 0.38±0.05      & 0.38±0.04      & 0.45±0.11            & 0.43±0.07           & \textbf{0.32±0.10}  \\
bracket matching-MD-1k                 & 0.20±0.02      & \textbf{0.06±0.09}  & 0.08±0.11      & 0.27±0.11      & 0.42±0.10            & 0.47±0.05           & 0.17±0.13      \\
bracket matching-MD-10k                & 0.16±0.02      & 0.14±0.13      & 0.07±0.11      & 0.19±0.11      & 0.23±0.14            & 0.37±0.14           & \textbf{0.03±0.01}  \\
bracket matching-MD-100k               & nan              & 0.11±0.11      & 0.20±0.16      & 0.19±0.13      & 0.11±0.07            & 0.14±0.14           & \textbf{0.02±0.01}  \\
chaotic forecast-SM-1k              & \textbf{0.00±0.00}  & 0.04±0.02      & 0.04±0.01      & 0.04±0.02      & 0.11±0.04            & 0.08±0.01           & 0.06±0.02      \\
chaotic forecast-SM-10k             & \textbf{0.00±0.00}  & 0.02±0.01      & 0.02±0.01      & 0.01±0.01      & 0.07±0.01            & 0.07±0.01           & 0.05±0.02      \\
chaotic forecast-SM-100k            & nan              & 0.02±0.02      & 0.03±0.01      & \textbf{0.01±0.01}  & 0.06±0.02            & 0.05±0.01           & 0.02±0.01      \\
chaotic forecast-MD-1k              & \textbf{0.00±0.00}  & 0.02±0.03      & 0.01±0.02      & \textbf{0.00±0.00}  & 0.11±0.02            & 0.10±0.00           & 0.03±0.02      \\
chaotic forecast-MD-10k             & \textbf{0.00±0.00}  & 0.01±0.02      & 0.02±0.01      & \textbf{0.00±0.00}  & 0.16±0.04            & 0.11±0.02           & 0.01±0.01      \\
chaotic forecast-MD-100k            & nan              & \textbf{0.00±0.01}  & 0.02±0.01      & \textbf{0.00±0.00}  & 0.10±0.02            & 0.09±0.01           & \textbf{0.00±0.00}  \\
c. pattern comp.-SM-1k    & 0.03±0.00      & \textbf{0.01±0.00}  & 0.02±0.01      & 0.04±0.03      & 0.07±0.02            & 0.08±0.01           & 0.03±0.01      \\
c. pattern comp.-SM-10k   & 0.04±0.01      & \textbf{0.01±0.00}  & \textbf{0.01±0.00}  & 0.02±0.00      & 0.07±0.01            & 0.07±0.01           & 0.02±0.00      \\
c. pattern comp.-SM-100k  & nan              & \textbf{0.01±0.00}  & \textbf{0.01±0.00}  & 0.02±0.00      & 0.07±0.01            & 0.07±0.01           & 0.02±0.01      \\
c. pattern comp.-MD-1k    & 0.08±0.00      & \textbf{0.01±0.00}  & 0.02±0.00      & 0.04±0.02      & 0.07±0.02            & 0.08±0.00           & 0.05±0.00      \\
c. pattern comp.-MD-10k   & 0.06±0.00      & \textbf{0.01±0.00}  & \textbf{0.01±0.00}  & 0.02±0.00      & \textbf{0.01±0.00}        & 0.07±0.02           & 0.03±0.01      \\
c. pattern comp.-MD-100k  & nan              & \textbf{0.01±0.00}  & \textbf{0.01±0.00}  & 0.02±0.00      & \textbf{0.01±0.00}        & 0.03±0.04           & 0.02±0.00      \\
c. postcasting-SM-1k           & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & 0.04±0.09            & 0.15±0.08           & 0.06±0.01      \\
c. postcasting-SM-10k          & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & \textbf{0.00±0.00}  \\
c. postcasting-SM-100k         & nan              & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & \textbf{0.00±0.00}  \\
c. postcasting-MD-1k           & 0.07±0.02      & \textbf{0.00±0.01}  & 0.01±0.01      & 0.15±0.04      & \textbf{0.00±0.00}        & 0.10±0.08           & 0.20±0.00      \\
c. postcasting-MD-10k          & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & 0.06±0.03      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.07±0.03      \\
c. postcasting-MD-100k         & nan              & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & 0.09±0.04      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & \textbf{0.00±0.00}  \\
cross situation-SM-1k                  & \textbf{0.01±0.01}  & 0.23±0.22      & 0.29±0.22      & 0.55±0.21      & 0.61±0.17            & 0.70±0.00           & 0.50±0.22      \\
cross situation-SM-10k                 & \textbf{0.01±0.01}  & 0.07±0.02      & 0.08±0.03      & 0.16±0.06      & 0.09±0.03            & 0.22±0.25           & 0.12±0.04      \\
cross situation-SM-100k                & nan              & \textbf{0.04±0.01}  & 0.08±0.01      & 0.14±0.04      & 0.06±0.01            & 0.13±0.20           & 0.05±0.01      \\
cross situation-MD-1k                  & 0.13±0.01      & \textbf{0.07±0.01}  & 0.10±0.01      & 0.56±0.18      & 0.52±0.27            & 0.80±0.00           & 0.57±0.20      \\
cross situation-MD-10k                 & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & 0.01±0.00            & 0.04±0.01           & \textbf{0.00±0.00}  \\
cross situation-MD-100k                & nan              & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & \textbf{0.00±0.00}  \\
d. pattern comp.-SM-1k      & \textbf{0.07±0.01}  & 0.17±0.04      & 0.24±0.15      & 0.17±0.12      & 0.41±0.09            & 0.46±0.09           & 0.13±0.02      \\
d. pattern comp.-SM-10k     & \textbf{0.07±0.01}  & 0.11±0.02      & 0.15±0.15      & 0.10±0.02      & 0.38±0.14            & 0.41±0.02           & 0.10±0.02      \\
d. pattern comp.-SM-100k    & nan              & \textbf{0.07±0.03}  & 0.11±0.02      & 0.09±0.01      & 0.22±0.18            & 0.41±0.02           & 0.09±0.02      \\
d. pattern comp.-MD-1k      & 0.57±0.06      & 0.59±0.03      & 0.69±0.11      & 0.63±0.16      & 0.63±0.21            & 0.80±0.02           & \textbf{0.41±0.05}  \\
d. pattern comp.-MD-10k     & 0.10±0.00      & 0.27±0.03      & 0.25±0.10      & 0.20±0.03      & \textbf{0.08±0.00}        & 0.56±0.26           & 0.20±0.01      \\
d. pattern comp.-MD-100k    & nan              & 0.29±0.02      & 0.30±0.03      & 0.19±0.03      & 0.08±0.01            & \textbf{0.00±0.00}       & 0.18±0.01      \\
d. postcasting-SM-1k             & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & 0.03±0.10            & 0.38±0.28           & 0.22±0.11      \\
d. postcasting-SM-10k            & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & \textbf{0.00±0.00}  \\
d. postcasting-SM-100k           & nan              & \textbf{0.00±0.01}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & \textbf{0.00±0.00}  \\
d. postcasting-MD-1k             & 0.63±0.01      & 0.64±0.04      & 0.65±0.09      & 0.74±0.07      & \textbf{0.00±0.00}        & 0.67±0.20           & 0.83±0.01      \\
d. postcasting-MD-10k            & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & \textbf{0.00±0.01}  & 0.30±0.06      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.54±0.03      \\
d. postcasting-MD-100k           & nan              & \textbf{0.00±0.00}  & \textbf{0.00±0.00}  & 0.12±0.03      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.01±0.01      \\
induction heads-SM-1k                  & \textbf{0.36±0.02}  & 0.60±0.09      & 0.62±0.08      & 0.65±0.04      & 0.40±0.34            & 0.64±0.04           & 0.53±0.09      \\
induction heads-SM-10k                 & 0.02±0.00      & 0.53±0.12      & 0.59±0.09      & 0.60±0.07      & \textbf{0.00±0.00}        & 0.18±0.29           & 0.46±0.07      \\
induction heads-SM-100k                & nan              & 0.49±0.13      & 0.48±0.08      & 0.58±0.04      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.44±0.03      \\
induction heads-MD-1k                  & 0.73±0.00      & 0.81±0.06      & 0.87±0.00      & 0.84±0.06      & \textbf{0.35±0.31}        & 0.77±0.05           & 0.74±0.02      \\
induction heads-MD-10k                 & 0.64±0.00      & 0.58±0.11      & 0.87±0.00      & 0.31±0.06      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.39±0.07      \\
induction heads-MD-100k                & nan              & 0.60±0.13      & 0.85±0.04      & 0.21±0.05      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.29±0.01      \\
selective copy-SM-1k                   & \textbf{0.48±0.04}  & 0.63±0.08      & 0.66±0.02      & 0.63±0.05      & 0.59±0.11            & 0.66±0.02           & 0.51±0.08      \\
selective copy-SM-10k                  & 0.44±0.02      & 0.54±0.16      & 0.65±0.02      & 0.55±0.13      & \textbf{0.37±0.27}        & 0.63±0.06           & 0.44±0.04      \\
selective copy-SM-100k                 & nan              & 0.61±0.11      & 0.65±0.01      & 0.52±0.14      & \textbf{0.10±0.05}        & 0.43±0.21           & 0.42±0.04      \\
selective copy-MD-1k                   & 0.75±0.01      & 0.65±0.13      & 0.87±0.00      & 0.78±0.14      & \textbf{0.58±0.18}        & 0.87±0.00           & 0.76±0.02      \\
selective copy-MD-10k                  & 0.70±0.00      & 0.54±0.25      & 0.87±0.00      & 0.32±0.35      & \textbf{0.04±0.07}        & 0.20±0.28           & 0.45±0.08      \\
selective copy-MD-100k                 & nan              & 0.54±0.24      & 0.88±0.00      & 0.11±0.16      & \textbf{0.01±0.00}        & 0.02±0.03           & 0.29±0.06      \\
simple copy-SM-1k                      & 0.47±0.01      & 0.64±0.06      & 0.67±0.01      & 0.65±0.04      & \textbf{0.26±0.33}        & 0.65±0.03           & 0.53±0.05      \\
simple copy-SM-10k                     & \textbf{0.00±0.00}  & 0.62±0.06      & 0.66±0.02      & 0.61±0.05      & \textbf{0.00±0.00}        & 0.12±0.26           & 0.49±0.03      \\
simple copy-SM-100k                    & nan              & 0.63±0.05      & 0.65±0.03      & 0.50±0.05      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.46±0.02      \\
simple copy-MD-1k                      & 0.76±0.00      & 0.82±0.05      & 0.87±0.00      & 0.81±0.02      & \textbf{0.04±0.09}        & 0.83±0.05           & 0.78±0.00      \\
simple copy-MD-10k                     & 0.70±0.00      & 0.72±0.03      & 0.87±0.00      & 0.50±0.08      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.69±0.03      \\
simple copy-MD-100k                    & nan              & 0.74±0.08      & 0.87±0.00      & 0.25±0.05      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.55±0.03      \\
sinus forecast-SM-1k                & \textbf{0.00±0.00}  & 0.01±0.00      & 0.02±0.00      & 0.02±0.01      & 0.16±0.12            & 0.31±0.21           & 0.01±0.00      \\
sinus forecast-SM-10k               & \textbf{0.00±0.00}  & 0.01±0.00      & 0.03±0.01      & 0.01±0.00      & 0.07±0.02            & 0.12±0.05           & 0.01±0.00      \\
sinus forecast-SM-100k              & nan              & 0.02±0.00      & 0.05±0.01      & \textbf{0.01±0.01}  & 0.05±0.05            & 0.06±0.03           & \textbf{0.01±0.01}  \\
sinus forecast-MD-1k                & \textbf{0.00±0.00}  & 0.05±0.01      & 0.03±0.01      & 0.04±0.02      & 0.35±0.27            & 0.18±0.12           & 0.05±0.01      \\
sinus forecast-MD-10k               & \textbf{0.00±0.00}  & 0.03±0.01      & 0.03±0.02      & 0.03±0.02      & 0.10±0.01            & 0.20±0.09           & 0.04±0.02      \\
sinus forecast-MD-100k              & nan              & 0.03±0.02      & 0.04±0.02      & \textbf{0.02±0.02}  & 0.08±0.01            & 0.10±0.02           & 0.04±0.02      \\
sorting problem-SM-1k                  & 0.51±0.01      & 0.49±0.03      & 0.51±0.01      & 0.61±0.05      & \textbf{0.08±0.21}        & 0.67±0.02           & 0.54±0.05      \\
sorting problem-SM-10k                 & 0.46±0.01      & 0.50±0.02      & 0.52±0.03      & 0.55±0.03      & \textbf{0.00±0.00}        & 0.06±0.16           & 0.31±0.02      \\
sorting problem-SM-100k                & nan              & 0.51±0.01      & 0.52±0.05      & 0.53±0.01      & \textbf{0.00±0.00}        & 0.05±0.16           & 0.22±0.08      \\
sorting problem-MD-1k                  & 0.75±0.00      & 0.75±0.01      & 0.77±0.01      & 0.70±0.07      & \textbf{0.00±0.00}        & 0.87±0.00           & 0.70±0.04      \\
sorting problem-MD-10k                 & 0.72±0.00      & 0.33±0.04      & 0.61±0.03      & 0.20±0.13      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.28±0.08      \\
sorting problem-MD-100k                & nan              & 0.24±0.05      & 0.70±0.01      & 0.02±0.02      & \textbf{0.00±0.00}        & \textbf{0.00±0.00}       & 0.11±0.04      \\
\bottomrule
\end{tabular}
}
\end{table}









\begin{table}
\caption{Résultats}
\label{tab:results}
\begin{tabular}{lccccccc}
\toprule
Tâches & ESN & GRU & LSTM & MAMBA & TRANS (DO) & TRANS (ED) & XLSTM \\
\midrule
adding problem-SM                 & \textbf{0.00}  & \textbf{0.00}  & 0.02      & 0.01      & \textbf{0.00}               & \textbf{0.00}              & \textbf{0.00}  \\
adding problem-MD                 & 0.69      & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}               & \textbf{0.00}              & \textbf{0.00}  \\
associative  rec.-SM             & 0.44      & \textbf{0.35}  & 0.57      & 0.59      & 0.36                   & 0.39                  & 0.40      \\
associative  rec.-MD             & 0.76      & 0.88      & 0.91      & 0.91      & \textbf{0.67}               & 0.78                  & 0.85      \\
bracket matching-SM               & 0.13      & 0.18      & 0.21      & 0.29      & 0.19                   & 0.27                  & \textbf{0.06}  \\
bracket matching-MD               & 0.13      & 0.01      & \textbf{0.00}  & 0.02      & 0.04                   & 0.01                  & \textbf{0.00}  \\
chaotic forecast-SM            & \textbf{0.00}  & \textbf{0.00}  & 0.01      & \textbf{0.00}  & 0.04                   & 0.04                  & 0.01      \\
chaotic forecast-MD            & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & 0.08                   & 0.07                  & \textbf{0.00}  \\
c. pattern completion-SM  & 0.02      & \textbf{0.01}  & \textbf{0.01}  & \textbf{0.01}  & 0.03                   & 0.06                  & \textbf{0.01}  \\
c. pattern completion-MD  & 0.06      & 0.01      & 0.01      & 0.02      & 0.01                   & \textbf{0.00}              & 0.01      \\
c. postcasting-SM         & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}               & \textbf{0.00}              & \textbf{0.00}  \\
c. postcasting-MD         & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & 0.02      & \textbf{0.00}               & \textbf{0.00}              & \textbf{0.00}  \\
cross situation-SM                & \textbf{0.00}  & 0.03      & 0.05      & 0.05      & 0.03                   & 0.04                  & 0.03      \\
cross situation-MD                & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}               & \textbf{0.00}              & \textbf{0.00}  \\
d. pattern completion-SM    & 0.06      & 0.06      & \textbf{0.05}  & 0.06      & 0.07                   & 0.36                  & 0.06      \\
d. pattern completion-MD    & 0.09      & 0.18      & 0.11      & 0.14      & 0.07                   & \textbf{0.00}              & 0.18      \\
d. postcasting-SM           & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}               & \textbf{0.00}              & \textbf{0.00}  \\
d. postcasting-MD           & \textbf{0.00}  & \textbf{0.00}  & \textbf{0.00}  & 0.08      & \textbf{0.00}               & \textbf{0.00}              & \textbf{0.00}  \\
induction heads-SM                & 0.01      & 0.35      & 0.37      & 0.48      & \textbf{0.00}               & \textbf{0.00}              & 0.39      \\
induction heads-MD                & 0.64      & 0.47      & 0.77      & 0.12      & \textbf{0.00}               & \textbf{0.00}              & 0.25      \\
selective copy-SM                 & 0.41      & 0.22      & 0.52      & 0.21      & \textbf{0.05}               & 0.17                  & 0.34      \\
selective copy-MD                 & 0.70      & 0.23      & 0.86      & \textbf{0.00}  & \textbf{0.00}               & 0.01                  & 0.23      \\
simple copy-SM                    & \textbf{0.00}  & 0.49      & 0.57      & 0.42      & \textbf{0.00}               & \textbf{0.00}              & 0.42      \\
simple copy-MD                    & 0.70      & 0.64      & 0.84      & 0.13      & \textbf{0.00}               & \textbf{0.00}              & 0.51      \\
sinus forecast-SM              & \textbf{0.00}  & 0.01      & 0.01      & 0.01      & 0.02                   & 0.02                  & 0.01      \\
sinus forecast-MD              & \textbf{0.00}  & 0.01      & \textbf{0.00}  & \textbf{0.00}  & 0.07                   & 0.07                  & 0.01      \\
sorting problem-SM                & 0.43      & 0.42      & 0.49      & 0.51      & \textbf{0.00}               & \textbf{0.00}              & 0.05      \\
sorting problem-MD                & 0.71      & 0.17      & 0.56      & \textbf{0.00}  & \textbf{0.00}               & \textbf{0.00}              & 0.06      \\
\bottomrule
\end{tabular}
\end{table}






