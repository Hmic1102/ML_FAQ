# Machine Learning Key Concepts

## Machine Learning

### Basic Concepts in Machine Learning

1. Overfitting / Underfitting  

   **Overfitting** refers to the phenomenon where a model fits the training data too well, capturing normal fluctuations, outliers, and noise as if they are actual features of the data. This leads to poor generalization performance on new data, showing excellent performance on the training set but performing poorly on validation/test sets.
    - Methods to mitigate overfitting include: 1) reducing the number of features; and 2) adding a **regularization term**. Regularization aims to reduce the influence of certain features on the predictive outcome. Common regularization terms include L1 and L2 regularization (see the regularization section for details).
    
    **Underfitting**, on the other hand, is the opposite of overfitting and refers to a model's lack of generalization ability.
    - Methods to address underfitting include: 1) increasing the number of training epochs; 2) increasing model features; and 3) reducing regularization.

2. Bias / Variance Trade-off

    **Bias** refers to the difference between the predicted result and the actual value, describing the model's fitting ability, while **variance** indicates the variability of model performance across different datasets, describing how sensitive the model is to data perturbations.
    
    Generally, simpler models have higher bias and lower variance, while more complex models have lower bias and higher variance, which corresponds to overfitting and underfitting.

    A common method to balance bias and variance is **cross-validation**. In k-fold cross-validation, the training set is split into $k$ parts, with one part serving as the validation set and the remaining $k-1$ parts as the training set for each iteration, repeated $k$ times so each subset serves as the validation set once. The average loss over $k$ runs gives the final estimate.

### Regularization in Machine Learning

1. L1 vs L2

    - **L1 Regularization**, also known as LASSO or L1 norm, is the sum of the absolute values of all parameters.
        $$
\lVert x \rVert_1 = \sum_{i=1}^m \lvert x_i \rvert
$$
    
    - **L2 Regularization**, also known as Ridge or L2 norm, is the square root of the sum of squares of all parameters.

        $$
            \lVert x \lVert_2=\sqrt{\sum_{i=1}^m x_i^2}
        $$

    - Both norms help reduce overfitting. L1 regularization can be used for **feature selection** as it encourages sparsity (explained below) but cannot be directly derived, so it doesn’t work with conventional gradient descent or Newton methods; common optimization methods include coordinate descent and Lasso regression. L2 regularization, however, is easily differentiable.

2. Why does L1 Regularization Lead to Sparsity?

    Compared to the L2 norm, L1 norm is more likely to yield a **sparse solution**, meaning that it can drive unimportant feature parameters to **0**.

    - Why?
    > Assume a loss function $L$ with respect to a parameter $x$, as shown in the diagram below. Here, the optimal point is at the red dot, where $x < 0$.
    >
    > ![l1vsl2_01](imgs/l1vsl2_01.jpg)
    >
    > Now, with L2 regularization, the new loss function $(L+Cx^2)$ is shown by the yellow line below, where the optimal $x$ lies at the blue dot. Although the absolute value of $x$ is reduced, it is still non-zero.
    >
    > ![l1vsl2_02](imgs/l1vsl2_02.jpg)
    >
    > With L1 regularization, the new loss function $(L+C\lvert x \lvert)$, shown in green below, has an optimal $x$ of 0.
    >
    > ![l1vsl2_03](imgs/l1vsl2_03.jpg)
    >
    > Through derivation, we can see that under L2 regularization, the loss function reaches a local minimum at $x=0$ only if the derivative of the original loss function is 0. However, with L1 regularization, it’s sufficient for the regularization coefficient $C$ to be greater than the derivative of $L$ for $x=0$ to be a local minimum.
    >
    > This analysis was for a single parameter $x$, but in reality, L1 regularization drives many parameters to 0, making the model sparse. This characteristic allows L1 regularization to be used for feature selection.

### Evaluation Metrics in Machine Learning

1. Precision / Recall / $F_1$ Score

    In binary classification problems, we commonly use precision, recall, and $F_1$ Score to assess the performance of a model. For any binary classifier, predictions can be categorized as follows:

    - **TP** (True Positive): correctly predicting the positive class.
    - **TN** (True Negative): correctly predicting the negative class.
    - **FP** (False Positive): incorrectly predicting a negative class as positive.
    - **FN** (False Negative): incorrectly predicting a positive class as negative.

    Based on the above definitions, we define the following metrics:

    - **Precision**:
        $$
            P=\frac{TP}{TP+FP}
        $$
        Precision reflects the proportion of correctly predicted positive samples out of all samples predicted as positive.
    
    - **Recall**:
        $$
            R=\frac{TP}{TP+FN}
        $$
        Recall indicates the proportion of correctly predicted positive samples out of all actual positive samples.

    - The **$F_1$ Score** is the harmonic mean of precision and recall:
        $$
            \frac{2}{F_1}=\frac{1}{P}+\frac{1}{R}
        $$
        $$
            F_1 = \frac{2 \times P \times R}{P + R} = \frac{2TP}{2TP+FP+FN}
        $$

2. Confusion Matrix

    A confusion matrix for classification results is shown below:

    ![ConfusionMatrix](imgs/ConfusionMatrix.jpg)

3. macro-$F_1$ vs micro-$F_1$

    Sometimes we encounter multiple binary confusion matrices (e.g., from multiple training and testing sessions, multiple datasets, or pairwise combinations in multiclass classification). We may want to assess model performance across these matrices.

    - **macro-$F_1$**

        One approach is to calculate precision and recall for each confusion matrix, then take the averages, yielding macro-$P$, macro-$R$, and the corresponding macro-$F_1$.
        $$
            \text{macro-}P = \frac{1}{n}\sum_{i=1}^n P_i, \qquad
            \text{macro-}R = \frac{1}{n}\sum_{i=1}^n R_i,
        $$
        $$
            \text{macro-}F_1 = \frac{2 \times \text{macro-}P \times \text{macro-}R}{\text{macro-}P + \text{macro-}R}
        $$
    
    - **micro-$F_1$**

        Another approach is to average the individual elements of each confusion matrix to obtain $\overline{TP}$, $\overline{TN}$, $\overline{FP}$, and $\overline{FN}$, then calculate micro-$P$, micro-$R$, and the corresponding micro-$F_1$ based on these values.
        $$
            \text{micro-}P = \frac{\overline{TP}}{\overline{TP}+\overline{FP}}, \qquad
            \text{micro-}R = \frac{\overline{TP}}{\overline{TP}+\overline{FN}},
        $$
        $$
            \text{micro-}F_1 = \frac{2 \times \text{micro-}P \times \text{micro-}R}{\text{micro-}P + \text{micro-}R}
        $$

4. ROC Curve / AUC

    **ROC Curve** (Receiver Operating Characteristic) and **AUC** (Area Under the ROC Curve) are popular metrics for imbalanced classification problems. To understand ROC, let’s introduce two terms based on the confusion matrix: True Positive Rate (TPR) and False Positive Rate (FPR).
    
    $$
        TPR = R = \frac{TP}{TP+FN}, \qquad
        FPR = \frac{FP}{TN+FP}
    $$
    
    Ideally, a model should have a high TPR and low FPR. For any trained model, we can calculate TPR and FPR on test data, plotting them as a point in ROC space. The ROC curve is created by plotting TPR against FPR at various threshold settings.

    ![ROC_02](imgs/ROC_02.png)

### Loss and Optimization

1. Convex Optimization Problems

    An optimization problem is a **convex optimization problem** if its objective function is a **convex function** and the feasible region is a **convex set** (any line segment between two points in the set lies entirely within the set).
    
    A function $f$ defined on a convex domain $\mathbb{D}$ is convex if and only if for any $x,y \in \mathbb{D}$ and $\theta \in [0,1]$,

    $$
        f(\theta x+(1-\theta)y) \le \theta f(x)+(1-\theta) f(y)
    $$

    ![convex_func](imgs/convex_func.jpg)

    Convex optimization problems are significant because any **local optimum** is also a **global optimum**. Thus, algorithms like greedy methods, gradient descent, and Newton’s method can solve convex optimization problems.

2. MSE / MSELoss

    **Mean Square Error** (MSE) is a common metric in regression tasks.

    $$
        E(f;D)=\sum_{i=1}^m (f(x_i) - y_i)^2
    $$

3. Is Logistic Regression with MSE Loss a Convex Optimization Problem?

    **No**. Logistic regression maps a linear model to a classification task through the non-linear sigmoid function. Its MSE is a non-convex function, so optimizing it may yield a local minimum, failing to reach a global optimum.

4. The Relationship between Linear Regression, Ordinary Least Squares, and Maximum Likelihood Estimation

    Common methods for solving linear regression include **Ordinary Least Squares (OLS)** and **Maximum Likelihood Estimation (MLE)**.

    - OLS minimizes the squared differences between predictions and true values.

        $$
            J(w)=\sum_{i=1}^m (h_w(x_i) - y_i)^2
        $$

    - MLE estimates parameters $h_w$ that maximize the probability of observing $x$ and $y$. Given that the error $\epsilon_i = y_i - h_w(x_i)$ follows a Gaussian distribution, the probability density function is:

        $$
            p(\epsilon_i) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(\epsilon_i)^2}{2\sigma^2}}
        $$

        Thus, the likelihood function is:

        $$
            \begin{aligned}
                L(h_w(x_i)) &= \prod_{i=1}^m p(y_i | h_w(x_i))\\
                &= \prod_{i=1}^m \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(y_i - h_w(x_i))^2}{2\sigma^2}} \\
            \end{aligned}
        $$

        Taking the log of both sides (log-likelihood) and removing constants yields the same form as the OLS formula, proving that both methods, though from different perspectives, produce identical results.

5. Relative Entropy and Cross-Entropy

    **Entropy** quantifies the uncertainty in a system, representing the expected information content.

    $$
        H(X) = -\sum_{x \in X}p(x_i) \log p(x_i)
    $$

    **Relative Entropy** or **Kullback-Leibler (KL) Divergence** measures the difference between two distributions $p$ and $q$.

    $$
        D_{KL}(p||q)=\sum_{x\in X} p(x)\log\frac{p(x)}{q(x)}=-\sum_{x\in X} p(x)[\log q(x) - \log p(x)]
    $$

    **Cross-Entropy** similarly measures the distance between distributions $p$ and $q$:

    $$
        CEH(p,q)=−\sum_{x \in X}p(x)logq(x)
    $$

    **Why Use Cross-Entropy instead of KL Divergence as a Loss Function?**
    
    Since $H(p)$ is constant when $p$ is known, cross-entropy and KL divergence are equivalent.

    ***In classification problems***, cross-entropy is commonly used as the loss function:

    - For binary classification:
        $$
            L = -\frac{1}{N} \sum_i [y_i \log(p_i) + (1 - y_i) \log (1 - p_i)]
        $$
    - For multi-class classification:
        $$
            L = -\frac{1}{N} \sum_i \sum_c^M [y_{ic} \log (p_{ic})]
        $$
        where $M$ is the number of classes.


### Naive Bayes

1. Probability Formulas and Bayes' Theorem

    - **Conditional Probability**: The probability of event A given that event B has occurred, denoted as $p(A|B)$;
    - **Joint Probability**: The probability of both events A and B occurring, denoted as $p(A, B) = p(A|B) * p(B)$;
    - **Total Probability**: If events $B_1, B_2, ..., B_n$ form a **complete set of events**, meaning they are mutually exclusive and cover the entire sample space, then for any event A:<br>$p(A)=\sum^n_{i=1} [p(A|B_i)*p(B_i)]$
    - **Bayesian Probability**: Often, it is difficult to directly find $p(A_i|B)$, but if $p(B|A_i)$, $p(A_i)$, and $p(B)$ are known, we can calculate:
        $$
            p(A_i|B) = \frac{p(B|A_i)*p(A_i)}{p(B)} = \frac{p(B|A_i)*p(A_i)}{\sum^n_{j=1}p(B|A_j)*p(A_j)}
        $$
        where $p(A_i|B)$ is the **posterior probability**, and $p(A_i)$ is the **prior probability**.


2. Naive Bayes Classifier

    - Let $x=\{a_1,a_2,...,a_m\}$ represent an item to be classified, where $a_i$ represents the feature attributes of $x$.
    - Define a set of classes $C=\{y_1,y_2,...,y_n\}$.
    - For each class $y_i$, compute the conditional probability for each feature attribute, i.e., $p(a_1|y_1)$, $p(a_2|y_1)$, ..., $p(a_m|y_1)$.
    - Using Bayes' theorem, calculate $p(y_i|x)=\frac{p(x|y_i)*p(y_i)}{p(x)}$.
    - Compute $p(y_i|x)$ for all classes, and the class with the highest probability $y_k$ is the predicted class.
    - This classifier is termed "Naive" Bayes because it assumes that all features in $x$ are **independent**.
    - Common Naive Bayes classifiers include:
        - GaussianNB: Assumes a **Gaussian distribution** for class priors, commonly used for continuous data.
        - MultinomialNB: Assumes a **multinomial distribution** for class priors, used for multinomial data.
        - BernoulliNB: Assumes a **Bernoulli distribution** for class priors, used for binary data.


3. Advantages and Disadvantages of Naive Bayes Classifier

    Main advantages:
    - Derived from classical mathematical principles, simple algorithm with stable classification efficiency;
    - Performs well with small datasets and is suitable for multi-class tasks;
    - Not sensitive to missing data, making it ideal for tasks like text classification;
    - Generally avoids overfitting.

    Disadvantages:
    - Assumes feature independence, which may not hold in reality;
    - Requires prior probability estimation, which, if inaccurate, may affect classification results;
    - Probability-based classification may sometimes be inaccurate.


4. Generative Model vs. Discriminative Model

    **Generative Models** learn the joint probability $P(X,Y)$, i.e., the probability of both the feature $x$ and the class $y$, and then calculate the conditional probability for each class. The predicted class is the one with the highest probability, where $P(Y|X) = \frac{P(X,Y)}{P(X)}$.
     - Generative models capture more information, such as the marginal distribution $p(x)$ for each feature;
     - Converge faster and perform well with small or sparse data;
     - Less prone to overfitting;
     - Generally, accuracy is lower than that of discriminative models.
    
    **Discriminative Models** learn the conditional probability directly, predicting the class $y$ given features $x$.
     - Offer flexible classification boundaries, able to fit more complex boundaries;
     - Only need to learn information relevant to classification, simplifying the problem;
     - Generally achieve higher accuracy than generative models.


### Support Vector Machine (SVM)

1. What is the core idea of SVM?

    SVM is a linear classifier in the feature space, with the objective of maximizing the **margin** between data points and the separating hyperplane. For linearly separable data, **hard margin maximization** is used to learn a linear hyperplane; for nearly linearly separable data, **soft margin maximization** with slack variables is applied; for non-linear data, kernel functions are used to map input data to a higher-dimensional feature space.

2. Hard Margin Maximization

    Define the **functional margin** of the hyperplane $(w,b)$ for a sample point $(x_i,y_i)$ as $\hat{\gamma}_i=y_i(wx_i+b)$. To avoid changes in functional margin by proportionally adjusting $w$ and $b$, we introduce the **geometric margin**, which is the perpendicular distance between the sample point and the hyperplane: $\gamma_i=y_i(\frac{w}{||w||}x_i+\frac{b}{||w||})$. The optimization goal of hard margin maximization is to find the hyperplane $(w,b)$ that maximizes the margin:
    $$
        \max_{w,b} \gamma, \text{s.t.,} y_i(\frac{w}{||w||}x_i+\frac{b}{||w||}) \geq \gamma, i=1,2,...,N
    $$
    This can be reformulated as:
    $$
        \min_{w,b} \frac{1}{2}||w||^2, \text{s.t.,} y_i(wx_i+b)-1 \geq 0,i=1,2,...,N
    $$
    Using the Lagrange multipliers, this problem can be converted to a dual problem, which makes it easier to solve:
    $$
        \min_\alpha \frac{1}{2}\sum_i^N \sum_j^N \alpha_i \alpha_j y_iy_j(x_i\cdot x_j)-\sum_i^N \alpha_i, \text{s.t.,} \sum_i^N \alpha_i y_i=0, \alpha_i \geq 0,i=1,2,...,N
    $$
    Converting to the dual problem also facilitates using **kernel functions** to address non-linear problems by replacing $(x_i\cdot x_j)$ with $K(x_i,x_j)=\phi(x_i)\cdot\phi(x_j)$.

3. Soft Margin Maximization

    In reality, data may not be linearly separable due to noise or outliers, which means that some sample points $(x_i,y_i)$ may not satisfy the functional margin constraint. To make the model more robust, we convert hard margins to soft margins. A slack variable $\xi_i$ is introduced for each sample point so that $y_i(wx_i+b)\geq1-\xi_i$.

    By adding a penalty parameter $C$, the objective function becomes $\frac{1}{2}||w||^2 + C\sum_i^N \xi_i$. The penalty parameter balances the goal of **maximizing margin while minimizing misclassified points**. When $\xi_i = 0$, the sample point is outside the margin; for $0 <\xi_i < 1$, the sample is correctly classified but within the margin boundary; and when $\xi_i > 1$, the sample is misclassified.

4. Hinge Loss

    The Hinge Loss function is shown below, with the x-axis representing the functional margin $\hat{\gamma}_i=y_i(wx_i+b)$. When $\hat{\gamma}_i\geq 1$, the classification is correct, and the loss is zero; when $\hat{\gamma}_i < 0$, the classification is incorrect; for $0 < \hat{\gamma}_i < 1$, the sample is correctly classified but within the margin boundary. Hinge Loss penalizes such samples, making SVM less dependent on a large number of training samples.

    ![hinge-loss](imgs/hinge-loss.jpg)


### Logistic Regression (LR)

1. What is the core idea of logistic regression?

    Logistic regression is mainly used for classification tasks. It assumes that a linear boundary can classify the data. Unlike linear regression, logistic regression primarily focuses on the relationship between the probability of classification and the input vector, i.e., the direct relationship between $P(Y=1)$ and $x$, and uses the probability to determine the class.

    Logistic regression primarily addresses binary classification. Given a dataset:

    $$
        D=(x_1,y_1), (x_2,y_2),...,(x_N,y_N), x_i\in R^n
    $$

    Since $w^T x+b$ is continuous, it can fit the conditional probability $p(Y=1|x)$. The ideal function is:

    $$
        p(Y=1|x)=
        \begin{cases}
            0, & z < 0\\
            0.5, & z = 0 \\
            1, & z > 0
        \end{cases}
        , z = w^T x + b
    $$

    This function is not differentiable, so we use the sigmoid function to approximate the probability:

    $$
        y=\frac{1}{1+e^z}, z=w^T x + b
    $$

    Viewing $y$ as the posterior class probability, we can rewrite the formula as:

    $$
        P(Y=1|x)=\frac{1}{1+e^z}, z=w^T x + b,\\
    $$
    
    $$
        z = \ln\frac{P(Y=1|x)}{1-P(Y=1|x)}
    $$

    Thus, logistic regression **uses the predicted values of a linear regression model to approximate the log odds of classification**. Its advantages include:

    - It can predict the probability of belonging to a class, which is useful for tasks that require probability estimation;
    - The log-odds function is a **convex function** in any order, making it solvable by various optimization algorithms.
  
2. Loss Function and Gradient in Logistic Regression

    Let: $P(Y=1|x) = p(x), P(Y=0|x) = 1 - p(x)$, so the likelihood function can be written as:

    $$L(w) = \prod[p(x_i)]^{y_i}[1-p(x_i)]^{1-y_i}$$
    
    Taking the logarithm (log-likelihood):
    
    $$
    \begin{aligned}
        l(w) = \ln L(w) & = \sum[y_i\ln p(x_i) + (1-y_i)\ln (1-p(x_i))] \\
        & = \sum[y_i\ln\frac{p(x_i)}{1-p(x_i)} + \ln(1-p(x_i))] \\
        & = \sum[y_i z_i - \ln(1 + e^{z_i})]
    \end{aligned}
    $$

    Using gradient descent, the gradient is:

    $$    
    \begin{aligned}
        \frac{\partial J(w)}{\partial w_j} & = -\sum_i \frac{\partial [y_i z_i - \ln(1 + e^{z_i})]}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_j} \\
        & = -\sum_i (y_i - p(x_i)) \cdot x_j
    \end{aligned}
    $$

    Updating weights:
    
    $$w_j := w_j + \eta(\sum_i (y_i - p(x_i)) \cdot x_j),\text{ for }i\text{ in range}(n)$$

3. Differences Between LR and SVM

    - Both focus on data points near the classification boundary. However, LR reduces the weights of data points far from the boundary, while SVM only considers misclassified points and those close to the boundary.
    - LR is a parametric model, assuming data distribution (e.g., Bernoulli) that may affect classification with imbalanced data, whereas SVM is non-parametric and independent of data distribution.
    - LR can produce probabilities, while SVM cannot directly produce probabilities.


### Decision Tree

1. What is a Decision Tree?

    A decision tree is a tree-like structure used for classification. Each node in the tree represents a decision, referred to as a **branch**. Data input to a decision tree is classified by following these branches until a terminal node or **leaf** is reached.

2. How to Determine the Classification Criteria?

    Generally, for each split, we select the feature that provides the most information for classification. This is evaluated by calculating the **entropy** of each feature.
    **Entropy** measures the disorder of a system, representing the expected information in all events.
    $$
        H(X) = -\sum_{x \in X}p(x_i) \log p(x_i)
    $$
    $$
        H(X|A) = -\sum_{i=1}^d p(A=a_i)H(X|A=a_i)
    $$
    Higher entropy means higher uncertainty. The basic concept in tree construction is to reduce entropy quickly as the tree depth increases. The faster entropy decreases, the shorter the resulting tree.

    First, we calculate the entropy $H(D)$ for the system without any feature. Then, we calculate the entropy of the system considering each feature. For each feature, we compute **information gain** $Gain(D|A) = H(D) - H(D|A)$ and choose the feature with the highest information gain.

3. Types of Decision Trees

    - **ID3**: Uses **information gain** to evaluate feature importance. However, it may incorrectly select sparse features with high information gain.
    - **C4.5**: Uses **information gain ratio** as a criterion, which is calculated as:
        $$
            GainRatio(D|A) = \frac{Gain(D|A)}{H(A)} = \frac{H(D)-H(D|A)}{H(A)}
        $$
        C4.5 addresses ID3’s issues with sparse features and can handle continuous and missing values.
    - **CART**: Uses **Gini coefficient** for classification tasks and **MSE** for regression tasks:
        $$
            Gini(X) = \sum_{x\in X} p(x_i)(1-p(x_i))
        $$
        $$
            Gini(X|A) = \sum_{i=1}^d p(A=a_i) Gini(X|A=a_i)
        $$

4. Random Forest (RF)

    Random forests use **bagging**, where $n$ samples are selected with replacement to build $m$ decision tree classifiers. These classifiers use **voting** to produce the final classification result. "Random" in random forests has two aspects:
    - Random sample selection (i.e., $n < \lVert D\rVert$) increases robustness to outliers and noise.
    - Random feature selection in each tree helps to filter out unimportant or irrelevant features.


### XGBoost

1. The Concept of Boosting

    When weak classifiers individually perform poorly, combining multiple weak classifiers can create a strong classifier. Boosting primarily focuses on reducing **bias**. Key differences between Boosting and Bagging include:
    - **Sample selection**: Bagging uses bootstrap sampling, while Boosting uses the same training set but adjusts sample weights based on the previous round’s results.
    - **Prediction**: Bagging gives equal weight to all classifiers, while Boosting assigns weights based on accuracy.
    - **Computation**: Bagging can train in parallel, while Boosting requires sequential training as sample weights depend on prior results.
    - *Bagging focuses on reducing variance; Boosting focuses on reducing bias.*

    > **Why Bagging reduces variance and Boosting reduces bias?**
    >
    > Bagging samples with replacement to train each model, effectively reducing overfitting and lowering variance.
    > Boosting sequentially optimizes weak classifiers to reduce bias.

2. XGBoost Basics

    In the $t$-th training round, **preserving results from previous rounds**, a new tree $f_t$ is added to **minimize the objective function**:
    $$
        \begin{aligned}
            Obj_t & = \sum_{i=1}^n l(y_i, \hat{y}_i^t) \\
            & = \sum_{i=1}^n l(y_i, \hat{y}_i^{t-1} + f_t(x_i)) \\
        \end{aligned}
    $$
    If the loss function is MSE, then the objective function becomes:
    $$
        \begin{aligned}
            Obj_t &= \sum_{i=1}^n (y_i - (\hat{y}_i^{t-1} + f_t(x_i)))^2 \\
            & = \sum_{i=1}^n[2(\hat{y}_i^{t-1} - y_i)f_t(x_i)+f_t(x_i)^2] + \sum_{i=1}^n ({y_i - \hat{y}_i^{t-1}})^2
        \end{aligned}
    $$
    Here, $\sum_{i=1}^n ({y_i - \hat{y}_i^{t-1}})^2$ is unrelated to the current round and can be treated as a constant. $(\hat{y}_i^{t-1} - y_i)$, often called the **residual**, represents the difference between the prediction from the previous round and the actual value. In XGBoost, each new round predicts this residual.

    ![xgboost](imgs/XGBoost.png)

    Additionally, XGBoost adds a regularization term on each leaf node $\Omega(f_t) = \gamma T + \lambda\frac{1}{2}\sum^T_{j=1} w_j^2$, where $T$ is the number of leaf nodes and $\lambda\frac{1}{2}\sum^T_{j=1} w_j^2$ is the L2 regularization term.




## Deep Learning

### Basic Concepts in Deep Learning

1. Why Do Neural Networks Need Bias Terms?

    For each neuron in a neural network, the function is defined as $y_i = W^TX_i + b$. This function essentially defines a decision boundary in the space. Without the bias term, the decision hyperplane would always pass through the origin, reducing the flexibility of the network. The bias term allows the network to better fit the data; without it, training may struggle to converge or may encounter other issues.

    > **Is it always necessary to use a bias term?**
    > 
    > No. For instance, after a convolutional layer, it is often better not to add a bias term if a Batch Normalization (BN) layer is used, as it would be ineffective and would take up GPU memory.
    > 
    > In BN, one critical step is:
    > 
    > $$
    >       \hat{x_i} = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}
    > $$
    > 
    > where $\mu_\mathcal{B}$ is the mean and $\sigma^2_{\mathcal{B}}$ is the variance. Since the bias term is canceled out in this calculation, it becomes ineffective in this context.


2. Back Propagation

    A BP (back-propagation) neural network consists of an input layer, an output layer, and several hidden layers. Input signals pass through the hidden layers and produce output from the output layer. The error between the output and the true value is computed, and this error is propagated back through the hidden layers to update the weights from the input layer. This backward flow of error, opposite to the training direction, is called "back propagation."

    Back propagation enables the use of gradient descent in neural networks by applying the chain rule to derive the error in hidden layers. It propagates the error back to obtain partial derivatives and allows optimization through gradient descent.


3. Gradient Vanishing and Gradient Explosion

    During back-propagation, if the gradient is consistently very small, the chain effect can lead to progressively smaller values as the error moves backward, causing the weights near the input layer to update minimally, slowing convergence. This is the **vanishing gradient** problem. Conversely, if gradients are too large, it can lead to **exploding gradients**, where values overflow, potentially resulting in NaN gradients.

    > When the activation function is Sigmoid, the vanishing gradient problem is more likely, as the maximum derivative of the Sigmoid function is only 0.25, as shown in the figure.
    > 
    > ![sigmoid](imgs/sigmoid.jpg)

    Common methods to mitigate vanishing/exploding gradients include:
    - Using alternative **activation functions** such as ReLU;
    - More appropriate **weight initialization** methods like Xavier or He initialization, which help maintain consistent variance during propagation;
    - **Batch Normalization (BN)**, which stabilizes $x$ distributions and can prevent gradient issues;
    - **Regularization** (L1, L2) for weights;
    - Using **ResNet**, where shortcut connections allow gradients to flow more easily across layers, reducing vanishing/exploding gradients;
    - **Gradient clipping** to manually prevent exploding gradients.


4. Can All Weights in a Neural Network Be Initialized to Zero?

    No. Initializing all weights to the same value would lead to identical updates during training, causing neurons to be indistinguishable from one another, effectively making the network function like a single neuron, which would limit its ability to fit data.

    Random initialization, or initialization methods such as Xavier or He, are typically used to prevent this.


5. Methods to Mitigate Overfitting in Deep Learning

    Common techniques to prevent overfitting in deep learning include:

    - Obtaining more and higher-quality data
      - Collecting new data
      - Data augmentation (e.g., mirroring, flipping images)
      - Using adversarial networks to generate data
    - Regularization (L1, L2)
    - Dropout
    - Early Stopping
    - Ensemble methods, such as Bagging and Boosting.


6. What is Dropout?

    Dropout randomly drops some connections between neurons during training. This prevents the model from relying too heavily on specific neurons, forcing it to learn more robust features and improving generalization.

    ![dropout](imgs/dropout.png)
    

7. Differences in Dropout and Batch Normalization During Training and Prediction

    Dropout is applied during training to reduce neuron dependencies, lowering the risk of overfitting. During prediction, dropout is turned off to use the fully trained model.

    In BN, each batch’s mean and variance are used for normalization during training. However, in testing, batch statistics may be unavailable (e.g., when predicting single samples). Therefore, during testing, the mean and variance calculated from all training data are used.

    > **Why not use the mean and variance of all training data during training?**
    > 
    > Using batch-specific statistics during training helps reduce overfitting by introducing variability that can improve model robustness.
    > 
    > Therefore, when using BN, the training set is usually shuffled and a large batch size is preferred to ensure each batch represents the dataset's distribution adequately.


8. Common Non-Linear Activation Functions and Their Advantages/Disadvantages

    Non-linear activation functions differentiate neural networks from simple perceptron networks, introducing non-linear characteristics into the network. Without non-linear activation, each layer would be a linear transformation of the previous layer, producing linear outputs no matter how deep the network is. Non-linear activation enables neural networks to approximate non-linear relationships.

   | Function Name |         Expression         |        Advantages           |      Disadvantages        |
   |:-------------:|:--------------------------:|:----------------------------|:--------------------------|
   |Sigmoid|$\displaystyle f(z)=\frac{1}{1+e^{-z}}$|1. Maps input to (0, 1) range.|1. Can lead to vanishing gradients; <br>2. Mean is not zero, leading to slower convergence.|
   |  tanh |$\displaystyle f(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$|1. Maps input to (-1, 1) range; <br>2. Solves non-zero mean issue of Sigmoid.|1. Still susceptible to vanishing gradients; <br>2. Exponential computation is costly.|
   |  ReLU |$f(z)=\max(0, x)$|1. Alleviates vanishing gradients; <br>2. Fast computation and convergence.|1. Mean is not zero; <br>2. **Dead ReLU Problem**: When $x<0$, gradient is zero, causing neurons to fail permanently. Solved by proper initialization or lowering the learning rate.|
   |Leaky ReLU|$f(z)=\max(\alpha \cdot x, x)$|1. Allows neurons to remain active in the negative range, reducing dead neurons.|1. Mean is not zero; <br>2. Choosing $\alpha$ can be tricky, usually set to a small value like 0.01.|


9. Gradient Descent vs. Stochastic Gradient Descent vs. Mini-Batch Gradient Descent

    |        Method       |                 Advantages          |            Disadvantages                |
    |:-------------------:|:-----------------------------------:|:---------------------------------------:|
    |Gradient Descent|1. Provides a stable direction for parameter updates;<br>2. Suitable for parallel computation.|1. Slow convergence during training; <br>2. Requires large memory for large datasets.|
    |Stochastic Gradient<br> Descent (SGD)|1. Performs updates based on a single random sample, making it computationally cheap;<br>2. Suitable for large datasets.|1. Requires more iterations; <br>2. Parameter updates can fluctuate significantly; <br>3. Not suitable for parallel computation.|
    |Mini-Batch Gradient<br> Descent (MBGD)|Combines the advantages of both: <br>1. Faster convergence than GD, more stable than SGD;<br>2. Leverages highly optimized matrix operations, suitable for parallelization.|1. Choosing a suitable learning rate can be challenging. Too low leads to slow convergence, too high causes excessive fluctuations. Using learning rate decay can help.|


10. Common Optimizers and Their Comparison

    |        Method       |                 Characteristics                                        |
    |:-------------------:|:-----------------------------------------------------------------------:|
    |GD / SGD / MBGD|1. Learning rate selection is challenging: too low leads to slow convergence, too high causes excessive fluctuations. <br>2. All parameters share the same learning rate, potentially underlearning sparse features.<br>3. Prone to saddle points, where the gradient is zero, causing training to stall.|
    | Momentum      |1. Adapts the concept of momentum from physics, maintaining inertia by retaining the previous update direction, enhancing stability and ability to escape local minima.<br>2. Observes the previous gradient, strengthening it if consistent with the current direction, weakening it if opposite.|
    |    AdaGrad    |1. Addresses the single learning rate limitation by adjusting the rate per parameter. Infrequent features receive a higher rate, while frequent ones receive a lower rate.<br>2. Works well for sparse data.|
    |    RMSprop    |1. An improvement on AdaGrad that averages historical gradients instead of accumulating them, slowing the decline of the learning rate over time.|
    |     Adam      |1. Combines advantages of AdaGrad and RMSprop, dynamically adjusting learning rates based on the moving average of the gradient and its variance.<br>2. Common default decay rates for two estimations are $\beta_1$ = 0.9 and $\beta_2$ = 0.999 in many ML libraries.|
    |    AdamW      |1. Corrects Adam’s issue where large gradients couldn’t be effectively regularized, by applying weight decay after regularization.|


11. How to Properly Use Transfer Learning?

    By using a pre-trained model that was trained on a large dataset, we can directly leverage its structure and weights for the task at hand.

    - Scenario 1: **Large** dataset, **high** similarity to the original dataset

        This is the ideal scenario. The best approach is to retain the original model structure and initial weights and then fine-tune the model on the new dataset.

    - Scenario 2: **Small** dataset, **high** similarity to the original dataset
        
        In this case, since the new data is similar to the original, retraining the model is unnecessary; it’s enough to adjust the output layer for the new task.

    - Scenario 3: **Large** dataset, **low** similarity to the original dataset

        Due to significant differences between the actual data and the pre-trained model’s data, using the pre-trained model may be inefficient. Instead, it’s better to retain only the model’s structure, reinitialize the weights, and start training from scratch.

    - Scenario 4: **Small** dataset, **low** similarity to the original dataset

        This is the most challenging situation. To prevent overfitting, we can’t train from scratch. Instead, we use the lower layers of the pre-trained model for feature extraction, compensating for the lack of data size, while training only the higher layers (typically more discriminative) for the new task. Thus, we **freeze** the weights of the first $k$ layers and train the last $n-k$ layers, adjusting the output layer for the new task.


## Natural Language Processing


### NLP 基本概念

1. 常见的文本相似度计算方法

    - 欧式距离，用于计算两等长**句子向量**的相似度。 $\text{distance} = \sqrt{(A-B)*(A-B)^T}$；
    - 余弦距离，用于计算两等长**句子向量**的相似度。 $\text{distance} = \frac{A*B^T}{|A|*|B|}$；
    - Jaccard 相似度。将句子看作单词的集合。则 A 与 B 的 Jaccard 相似度为：$\text{similarity} = \frac{|A\cap B|}{|A\cup B|}$；
    - TF-IDF。TF 是词频 (Term Frequency)，表示在一个文章中某个单词出现的频率；IDF 是逆文本频率指数 (Inverse Document Frequency)，表示含有该关键词的文档占所有文档的比例。TF-IDF 建立在以下假设上：对区别文档最有意义的词语应该是那些在**文档中出现频率高**，而在整个文档集合的**其他文档中出现频率少**的词语；
    - 最小编辑距离。一种经典的距离计算方法，用来度量字符串之间的差异。将字符串 A 不断修改（增删改），直至成为字符串 B，所需要的修改次数代表了字符串 A 和 B 的差异大小。常使用动态规划来计算最小编辑距离。


2. word2vec 模型

    在 NLP 中，我们希望用一个数学形式表示不同的单词，于是便有了词向量。最初的词向量是 one-hot 词向量，但这种向量维度过大，非常稀疏，且不能反映词与词之间的关系。于是便有了**分布式词向量**，即固定 embedding 的维度，embedding 中的每一个值都是通过计算不同单词的贡献得到的。

    训练 word2vec 模型主要有两种方式：CBOW 和 Skip-Gram。
    - CBOW 是让模型根据某个词前面的 C 个词和之后的 C 个词，预测这个词出现的概率。如图，训练过程其实就是学习这两个矩阵 $W$ 和 $W'$，其中，$W$ 矩阵又被叫做 lookup table，即所有词嵌入向量的词表。

        ![word2vec-CBOW](imgs/word2vec-CBOW.jpg)
    - Skip-Gram 和 CBOW 相反，是根据某一个词来预测它的前 C 个词和后 C 个词。同样训练两个矩阵 $W$ 和 $W'$，其中，$W$ 矩阵是 lookup table。一般来说，Skip-Gram 的训练时间比 CBOW 要慢。

        ![word2vec-skip-gram](imgs/word2vec-skip-gram.jpg)
    
    为了加快训练速度，word2vec 采用了两种优化方式。
    - Hierarchical Softmax，用霍夫曼树代替神经网络，本质上是将 n 分类变成 log(n) 次二分类。
    - Negative Sampling，由于霍夫曼树中高频词离根结点较近，但是如果中心词是较生僻的词，那么就要消耗很长时间。简要来说就是从负样本中选取一部分来更新，而不是更新全部的权重。


3. GloVe 模型

    GloVe 模型利用了词语的共现频率来计算相关性。首先引入词语的共现矩阵 $X$，其中 $X_{ij}$ 是在 word i 的上下文中 word j 的出现次数，$X_i = \sum_k X_{ik}$ 是出现在 word i 的上下文中所有词的出现次数，则共现概率为 $P_ij = P(j|i) = \frac{X_{ij}}{X_i}$，是word j 出现在 word i 上下文的概率。可以发现，共现概率的比例可以反映两个词的**相关度**。


### HMM / CRF

1. 隐马尔可夫模型 Hidden Markov Model, HMM

    - 马尔可夫链，即一个状态序列，满足在任意时刻 $t$ 的状态仅与其前一时刻的状态有关。隐马尔可夫链，则是无法直接观测到某一时刻的状态，而是要通过其他的观测状态才能预测隐藏的状态。
    - 隐马尔可夫模型的两个基本假设：
      - **齐次性假设**：即隐藏的马尔可夫状态在任意时刻的状态只依赖于前一时刻的状态，与其他时刻的状态无关；
      - **观测独立性假设**：任意时刻的观测状态只取决与当前状态的隐藏状态，和其他时刻的观测状态或隐藏状态无关。
    - 隐马尔可夫模型的五个要素：
      - **隐藏状态集** $Q$ = {$q_1$, $q_2$, ..., $q_N$}，即隐藏节点只能取值于该集合中的元素。
      - **观测状态集** $V$ = {$v_1$, $v_2$, ..., $v_M$}，即观测节点的状态也只能取值于该集合中的元素。
      - **隐藏状态转移矩阵** $A$ = $[a_{ij}]_{N\times N}$，表示从一种隐藏状态到另一种隐藏状态的转移概率。
      - **观测概率矩阵** $B$ = $[b_{ij}]_{N\times M}$，表示对于某一种隐藏状态，其观测状态的分布概率。
      - **初始隐藏状态概率** $\pi$ = $[p_1, p_2, ..., p_n]$，表示初始时刻处于各个隐藏状态的概率。
    - 隐马尔可夫模型要求解的基本问题：
      - **概率计算问题**。对于已知模型 $\lambda$ = $(A, B, \pi)$，和已知观测序列 $O$ = {$o_1$, $o_2$, ..., $o_M$}，求产生这种观测序列的概率是多少，即求 $p(O|\lambda)$。
      - **学习问题**。对于已知观测序列 $O$ = {$o_1$, $o_2$, ..., $o_M$}，求解模型 $\lambda$ = $(A, B, \pi)$ 的参数，使得产生这种观测序列的概率 $p(O|\lambda)$ 最大，即用**最大似然估计**方法估计模型的参数。
      - **解码问题**。同样对于已知模型 $\lambda$ = $(A, B, \pi)$，和已知观测序列 $O$ = {$o_1$, $o_2$, ..., $o_M$}，求解最优的隐藏状态序列 $I$ = {$i_1$, $i_2$, ..., $i_N$}，使得 $p(I|O)$ 最大。
    - 对于基本问题的解法：
      - 对第一个问题的解法：
        - 暴力解法：时间复杂度为 $O(TN^T)$；
        - 前向算法：利用动态规划思想，将前面时刻计算过的概率保存下来。
          - 对于第一个时刻的状态：$a_1(i) = \pi_ib_i(o_1)$, $i\in [1,N]$；
          - 对于第 $t$ 个时刻的状态：$a_t(i) = [\sum_{j=1}^N a_{t-1}(j)a_{ji}]b_i(o_t)$。
      - 对第二个问题的解法：
        
        Baum-Welch 算法：与 EM 算法相似，在 E-step 中，计算联合分布 $P(O,I|\lambda)$ 和条件分布 $P(I|O,\bar{\lambda})$，根据联合分布和条件分布计算期望表达式 $L(\lambda,\bar{\lambda})$；在 M-step 中最大化 $\lambda$ 的值，使得 $\bar{\lambda} = \argmax_\lambda L(\lambda,\bar{\lambda})$。
    
      - 对第三个问题的解法：

        Viterbi 维特比算法：可以看作一个求最长路径的动态规划算法。

        初始化两个状态变量：$\delta_t(i)$ 表示在 $t$ 时刻隐藏状态为 $i$ 的所有状态转移路径中概率最大值，初始化 $\delta_1(i) = \pi_i b_i(o_1)$。$\psi_t(i)$ 则是在 $t$ 时刻使得隐藏状态为 $i$ 的转移路径中概率最大的前一时刻的隐藏状态，初始化为 0。则两状态变量的递推表达式为：

        $$
            \begin{cases}
                \delta_t(i) = \max_{1\leq j\leq N}[\delta_{t-1}(j)a_{ji}]b_i(o_t) \\
                \psi_t(i) = \argmax_{1\leq j\leq N}[\delta_{t-1}(j)a_{ji}]
            \end{cases}
        $$

        在第 $T$ 步，取 $\delta_T(i)$ 最大值即为最可能隐藏序列出现的概率，此时最大的 $\psi_T(i)$ 即为第 $T$ 的状态。

        随后，从第 $T$ 步开始回溯，即 $i^*_{t-1}$ = $\psi_t(i^*_t)$，得到完整的隐藏序列 $I=(i^*_1, i^*_2, ..., i^*_T)$。


2. 条件随机场 Conditional Random Field, CRF

    首先介绍随机场。一组随机变量，分布在同一个样本空间，那么它们就组成了一个随机场。我们希望利用这些随机变量之间的关系来解决实际问题。

    马尔可夫随机场是满足马尔可夫独立性的随机场，即每个节点仅与其相邻的节点有关系。并不像贝叶斯网络（有向无环图）一样，通过变量之间的条件分布建模（节点与节点之间的依赖关系），马尔可夫随机场是根据变量之间的联合分布来建模的。当知道了变量之间的联合分布，则它们之间的条件分布也就能求解出来。因此，马尔可夫随机场是**生成模型**。

    条件随机场则是对条件概率建模。即已经观测到部分点的前提下，求解整张图的分布。

    HMM 是通过对可观测序列和隐藏序列的联合概率建模，估计模型的隐含变量的分布，再计算概率最大的隐藏序列，是一个**生成模型**。CRF 则是直接对条件概率 $P(I|O)$ 建模，通过可观测序列直接判别出隐藏序列，是一个**判别模型**。

    比较常见的条件随机场是线性链条件随机场。设 $X = (X_1, X_2, ..., X_n)$, $Y = (Y_1, Y_2, ..., Y_n)$ 均为线性链表示的随机变量序列。在给定 $X$ 的情况下，随机变量序列 $Y$ 的条件概率分布构成线性链条件随机场，即 $Y$ 的分布只与其相邻的节点有关。

    与 HMM 相似，条件随机场主要求解的也是三个问题：
     - 概率计算问题：给定条件随机场 $P(Y|X)$, 观测序列 $x$ 和隐藏序列 $y$，计算条件概率 $P(Y_i=y_i|x)$，可以通过前向后向解法求解
     - 学习问题：已知观测序列和隐藏序列，通过极大似然估计法来学习模型的最大概率参数。
     - 预测问题：给定条件随机场 $Y=(Y|X)$ 和观测序列 $x$，求条件概率最大的隐藏序列 $y^*$，即**对观测序列进行标注**。预测问题的常用算法是维特比算法。

    CRF 是一个序列化标注算法，接收一个输入序列 $X = (x_1, x_2, ..., x_n)$，输出目标序列 $Y = (y_1, y_2, ..., y_n)$，可以被看作是一个 Seq2Seq 模型。在词性标注任务中，输入为文本序列，输出则为对应的词性序列。

    相比于 HMM 需要对状态转移矩阵和观测概率矩阵建模，CRF 属于判别模型，其直接对 $P(I|O)$ 建模：

    $$ 
        P(I|O) = \frac{1}{Z(O)}e^{\sum_i^T \sum_k^M \lambda_k f_k(O, I_{i-1}, I_i, i)}
    $$

    其中，下标 i 表示当前所在的节点（token）位置，下标 k 表示第 k 个特征函数，并且每个特征函数都附属一个权重 $\lambda_k$，$\frac{1}{Z(O)}$ 是归一化系数。


### RNN / LSTM

1. 为什么需要 RNN？

    循环神经网络（Recurrent Neural Network, RNN）。当给定的数据是序列型数据，如文本、音频等数据时，我们往往希望模型能够学习到给定数据的上下文环境。例如，对于一个句子，序列模型试图从同一个句子前面的单词推导出关系。

    ![RNN](imgs/rnn.png)

    在循环神经网络的每一个时间步骤（time step）中，我们取一个输入 $x_i$ 和上一个节点的权值 $h_{i-1}$ 作为输入，并产生一个输出 $y_i$ 和权值 $h_i$，这个权值又被传递到下一个时间步骤，直到输入序列被读取完毕。

    ![multi-tasks-rnn](imgs/multi-tasks.jpg)

    普通的 RNN（Vanilla RNN）常使用 BP 算法来训练权值，但由于**梯度消失 / 梯度爆炸**问题，RNN 会丧失学习远距离信息的能力。为了解决远距离依赖问题，提出了 LSTM（Long Short-Term Memory）。


2. LSTM 网络

    LSTM（Long Short-Term Memory）相对于普通 RNN 网络，能够显著的缓解长期依赖关系丢失的问题。LSTM 的主要思想是利用**门结构**来去除或添加单元之间信息传递的能力。LSTM 拥有三个门，来保护和控制单元状态，分别为**遗忘门**、**输入门**和**输出门**。

    - 遗忘门

        ![forget-gate](imgs/forget-gate.png)
        第一步是决定从上一个单元中保留多少消息。将上一单元的状态 $h_{t-1}$ 和这一层的输入 $x_i$ 经过 sigmoid 层，输出一个 0-1 的值，代表要从上一层的单元状态保留多少信息。

    - 输入门

        ![input-gate](imgs/input-gate.png)
        这一步是决定在这一层的单元状态中保留多少信息。将上一单元的状态 $h_{t-1}$ 和这一层的输入 $x_i$ 分别经过 sigmoid 层和 tanh 层，得到一个候选的单元状态 $\tilde{C}_t$。

        ![update-cell](imgs/update-cell.png)
        随后，根据遗忘门得到的遗忘比例 $f_t$ 和这一层要输入的单元状态 $\tilde{C}_t$，得到这一层的最终单元状态 $C_t = f_t*C_{t-1} + i_t*\tilde{C}_t$。

    - 输出门

        ![output-gate](imgs/output-gate.png)
        最终，我们需要决定这一层的单元的输出状态。将上一单元的状态 $h_{t-1}$ 和这一层的输入 $x_i$ 经过 sigmoid 层，确定要输出的部分 $o_t$，再将这一层的单元状态 $C_t$ 经过 tanh 层，再与 $o_t$ 结合，得到最终的输出状态 $h_t$。


3. GRU 网络

    与 LSTM 对比，GRU 网络更加简单，训练更加高效。GRU 去除了单元状态，将 LSTM 的 3 个门减少到 2 个，分别为更新门和重置门，分别决定了应该让多少信息通过单元，以及应该丢弃多少信息。

    ![gru](imgs/gru.jpg)


4. 如何计算 LSTM 和 GRU 的参数量？

    一个单元内一共有四个非线性门 ($W[h_{t-1},x_t] + b$)，每一个门内的可训练变量包括一个矩阵 $W$ 和一个置项 $b$。

    因此，一个 LSTM 非线性门的参数即为 **(embed_size + hidden_size) * hidden_size +hidden_size**，LSTM 四个门的总参数量为 **((embed_size + hidden_size) * hidden_size +hidden_size) * 4**。

    同理，一个 GRU 单元的参数量为 **((embed_size + hidden_size) * hidden_size + hidden_size) * 3**。


4. RNN / LSTM 的局限性
   
   - 对于 RNN 来说，在访问一个单元前需要遍历之前所有的单元，使得在长文本下极度容易出现梯度消失问题。
   - LSTM 利用门机制稍微缓解了 RNN 的梯度消失问题，但在超长文本前仍然存在该问题。
   - 在 LSTM 中，在每一个单元中都有 4 个 MLP 层，需要消耗大量的计算资源，且模型本身不利于并行化。


### TextCNN

1. 如何卷积？

    输入一个长度为 $s$ 的句子，将其分词后映射到词向量，假设词向量的维度为 $d$，那么该句子可以表示为一个 $s\times d$ 的矩阵。将该矩阵看作一张图像，用卷积神经网络提取特征。

    **文本卷积和图像卷积的区别在与文本序列只在垂直方向做卷积**，即卷积核的宽度固定为词向量的维度 $d$。 


2. TextCNN 的优缺点？

    - 优点：网络结构简单，训练速度快，适合进行并行化，对短文本效果好；使用 Max-Pooling，便于提取最关键信息，因此适用于文本分类等任务。
    - 缺点：全局 Max-Pooling 丢失了结构信息，很难发现文本中的依赖关系；只能学习到关键词是什么，无法学习到关键词的频率和顺序。


### Attention Mechanism / Transformer

1. Seq2Seq 中的 Attention 机制

    在 Seq2Seq 中，我们使用 encoder 将输入文本转化为一个定长向量，再用 decoder 将该向量转换为输出文本。但是，在面对长文本时，我们很难在定长向量中保留完整的输入文本信息，因此在 decode 时会存在信息丢失的问题。为了缓解这个问题，我们引入了 Attention 机制。

    以机器翻译任务举例，在翻译到某一个单词时，我们希望能够注意到这个单词所对应的上下文，并结合之前已翻译的部分作出相应的翻译。这样，我们在 decoder 中就可以注意到输入文本的全部信息，而不只局限于那个定长的向量。

    Attention 的计算过程如下：

    - 得到 encoder 中的 hidden state $\overrightarrow{h_e} = (h_1, h_2, ..., h_n)$。
    - 假设当前翻译到的 decoder state 为 $\overrightarrow{s_{t-1}}$，则可以计算该状态与输入的每一个单元 $h_j$ 状态的关联性 $e_{tj} = a(s_{t-1},h_j)$，写成向量形式则为 $\overrightarrow{e_t} = a(\overrightarrow{s_{t-1}}, \overrightarrow{h})$，其中，$a$ 是相关性的计算，常见的计算方式有：
      - 直接点乘 $a(s_{t-1},h_j)=\overrightarrow{s_{t-1}}^T\cdot\overrightarrow{h}$；
      - 加权点乘 $a(s_{t-1},h_j)=\overrightarrow{s_{t-1}}^T\cdot W \cdot\overrightarrow{h}$，其中，$W$ 是可训练矩阵；
      - 多层感知机 $a(s_{t-1},h_j)=V \cdot \tanh(W_1 \cdot \overrightarrow{s_{t-1}} + W_2 \cdot \overrightarrow{h})$，其中，$V$、$W_1$、$W_2$ 都是可训练矩阵；
      - 缩放的点乘 $a(s_{t-1},h_j)=\frac{\overrightarrow{s_{t-1}}^T\cdot\overrightarrow{h}}{\sqrt{|\overrightarrow{h}|}}$。Softmax 函数对非常大的输入很敏感。这会使得梯度的传播出现问题，并且会导致学习的速度下降，甚至会导致学习的停止。那如果我们使用 $\sqrt{|\overrightarrow{h}|}$ 来对输入的向量做缩放，就能够防止进入到 softmax 函数的饱和区，使梯度过小。
    - 对 $\overrightarrow{e_t}$ 进行 softmax 操作后得到 Attention 分布 $\overrightarrow{\alpha_t} = softmax(\overrightarrow{e_t})$，其中，$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{i=1}^n \exp(e_{ti})}$。
    - 计算得到**上下文表示** $\overrightarrow{c_t}=\sum_{j=1}^n \alpha_{tj}\cdot h_j$。
    - 我们可以将该上下文表示利用到下一个时间步的状态生成 $s_t = f(s_{t-1}, y_{t-1}, c_t)$。


2. Q(Query), K(Key), V(Value)

    在 Attention 中，Q(Query) 指的是被查询的向量，即根据什么来关注其他的单词；K(Key) 指的是查询的向量，即被关注的向量的关键词；V(Value) 则是的被关注的信息本身。

    使用 Q 和 K 计算了相似度之后得到相似度评分，之后有了相似度评分，就可以把内容 V 加权回去了。


3. Transformer

    既然我们知道 Attention 机制本身就可以获取上下文信息，那么我们可不可以将原本的 RNN 结构完全去掉，仅仅依赖于 Attention 模型呢？这样我们可以使得训练并行化，并且可以拥有全局的信息。根据这个思想，产生了 Transformer 模型。其模型结构如下：

    ![transformer](imgs/transformer.jpg)

    - Self-Attention 机制
    
        Seq2Seq 中的 Attention 机制是在 decode 过程中，逐步计算对应的上下文表示，仿照这个思想，Self-Attention 就是在 encode 阶段，便考虑到每个输入单词与其他单词的关联性，从而得到具有上下文信息的 input embedding 信息。因此，对于 Self-Attention，其 Q, K, V 都来自于同一个输入矩阵，即 Q=K=V。

        Self-Attention 的计算过程如下：
      - 输入序列 $\overrightarrow{x}$；
      - 将 $\overrightarrow{x}$ 分别与对应 Q, K, V 的三个可训练矩阵 $W_q$, $W_k$, $W_v$ 点乘，得到 $Q=\overrightarrow{x}\cdot W_q$, $K=\overrightarrow{x}\cdot W_k$, $V=\overrightarrow{x}\cdot W_v$；
      - 计算 $Attention(Q,K,V)=softmax(\frac{Q\cdot K^T}{\sqrt{d_K}})\cdot V$，其中，$d_K$ 为 $K$ 的维度。

    - Multi-Head Attention

        为了使模型能够**从不同角度获取输入序列的上下文信息表示**，同时引入多组 ($W_{qi}$, $W_{ki}$, $W_{vi}$) 矩阵，分别得到多个 ($Q_i$, $K_i$, $V_i$)，再将它们**按列拼接**，之后经过一个联合矩阵 $W_o$，得到最终的 Attention 表示。过程如图所示：

        ![multi-head](imgs/multi-head.jpg)

        注意，在 Transformer 的模型中，有多个 Multi-Head Attention 步骤。其中，encoder 中的 Attention 和 decoder 中的第一步 Attention 的步骤都仅以前一级的输出作为输入，而在 decoder 中的第二步 Attention 则不仅接受来自前一级的输出，还要接收 encoder 的输出。

        即，在第一种 Multi-Head Attention 中，有 $Q = K = V$，在第二种 Multi-Head Attention 中，则 $Q \neq K = V$: $Q$ 指的是 target 序列，而 $Q$ 和 $K$ 指的是输入序列。

    - Positional Encoding

        由于 Transformer 模型没有循环结构或卷积结构，为了使模型能够学习到输入序列的顺序，我们需要插入一些关于 tokens 位置的信息。因此提出了 **Positional Encoding** 的概念，其与 input embedding 具有相同的维度，便于相加。

        但是，如果直接使用计数的方式来进行 encoding，即 $pos = 1, 2, ..., n - 1$，那么最后一个 token 的encoding 将会比第一个 token 大很多，与原 embedding 相加后会造成数据不平衡的现象。原论文作者们的方法是使用了不同频率的正弦和余弦函数来作为位置编码：
        $$
            \begin{aligned}
                PE_{(pos,2i)}   & = sin(pos/10000^{2i/d_{model}}) \\
                PE_{(pos,2i+1)} & = cos(pos/10000^{2i/d_{model}}) \\
            \end{aligned}
        $$

        ```python
        def get_positional_embedding(d_model, max_seq_len):
            positional_embedding = torch.tensor([
                    [pos / np.power(10000, 2.0 * (i // 2) / d_model) for i in range(d_model)]  # i 的取值为 [0, d_model)
                    for pos in range(max_seq_len)]  # pos 的取值为 [0, max_seq_len)
                )
            # 进行 sin / cos 变换
            positional_embedding[:, 0::2] = torch.sin(positional_embedding[:, 0::2])
            positional_embedding[:, 1::2] = torch.cos(positional_embedding[:, 1::2])
            return positional_embedding
        ```
    
    - Add & Norm 层
      - Add 指的是 Residual Connection，$y=F(x)+x$. 与 ResNet 的原理相似，是将上一层的信息直接传到下一层，可以帮助解决多层神经网络训练困难的问题。同时，引入残差连接有助于减轻神经网络在深层退化的问题。
      - Norm 指的是 Layer Normalization，在层与层之间对每一行数据进行缩放。这样可以缓解梯度消失的状况，同时使模型更快收敛。
        > **Batch Normalization 和 Layer Normalization 的区别？**
        > 
        > 在 BN 中，我们将每一个 batch 中的数据**按列**进行缩放。而在 NLP 任务中，由于输入序列的长度是不确定的，且不同行同一位置的单词直接并没有直接联系，直接做缩放可能会影响原语义表达。因此，在 NLP 等序列型任务中，我们一般采用 Layer Normalization，即对每一行数据进行缩放。
    

4. BERT: Bi-directional Encoder Representation from Transformers

    - 双向表示

        区别于 Bi-LSTM 的双向表示，分别正序和反序得到表示再进行拼接，BERT 中的双向指的是根据前文和后文来预测被 masked 的单词。

    - Embedding

        BERT 中的 embedding 由三个部分组成：Token Embedding，Segment Embedding，Position Embedding。
        - Token Embedding 是词向量，其中，第一个词为 [CLS] 标记，可以用于之后的下游任务。
        - Segment Embedding 用于区分 BERT 输入的两个句子，之后的 pre-training 将会用到。
        - Position Embedding 由学习得到，而不是普通 Transformer 中的三角函数。

    - Pre-training Tasks
        - Masked LM
            
            在训练过程中，将 15% 的单词用 [mask] 代替，让模型去预测被遮挡的单词，最终的损失函数只计算被遮盖的 token。

            但是如果一直用 [mask] 表示（实际预测时并不会遇到 [mask] 标记）会影响模型，因此作者设置了一下规则：80% 的时间用 [mask] 来代替被遮盖的单词，10% 的时间随机用另一个单词代替，剩下 10% 的时间保留原单词。

            值得注意的是，模型并不知道哪些单词被遮盖了，这使得模型能够关注到每一个单词，依赖上下文信息预测单词，赋予了模型一定的纠错能力。
        
        - Next Sentence Prediction

            对于输入的两个句子 A 和 B，让模型预测 B 是否应该是 A 的后一句。该任务的目的是让模型理解两个句子直接的关系。
    
    - 为什么BERT在第一句前会加一个 [CLS] 标志?

        为了获得整个句子的语义表示，用于其他任务。一个没有明显语义的 [CLS] 标记会更加**公平**地融合句子中每个单词的语义，从而获得更加完整的句子表示。

    - BERT 的优缺点？

        优点是建立在 Transformer 上，相对rnn更加高效，具有强大的信息提取能力，能捕捉更长距离的依赖。且双向模型比单向的 Transformer 效果更好；
        
        缺点则是该模型几乎无法修改，只能拿来直接用。由于只能预测 15% 的词，模型收敛较慢，需要强大算力支撑。

    - 使用BERT预训练模型为什么最多只能输入 512 个词，最多只能两个句子合成一句？

        这是由于在预训练的时候，在参数中设置了 position embedding 的大小和 segment embedding 的大小，分别为 512 和 2。在这之外的单词和句子会没有与之对应的 embedding。

    - BERT 的输入和输出分别是什么？

        输入是 token embedding，segment embedding 和 position embedding，输出是文本中各个字 / 词融合了全文语义信息后的向量表示。

    - 计算 BERT 模型的参数数量？
        - 词向量参数：vocab_size=30522, hidden_size=768, max_position_embedding=512, token_type_embedding=2，因此参数量为 (30522 + 512 + 2) * 768。
        - Multi-head Attention：len = hidden_size = 768, $d_k$ = $d_q$ = $d_v$ = $d_{model}/n_{head}$ = 768 / 12 = 64，将12个头进行拼接后还要进行线性变换，因此参数量为 768 * 64 * 12 * 3 + 768 * 768。
        - 前馈网络参数：$\text{FFN}(x)=\max(0, xW_1+b_1)W_2 + b_2$，W_1 和 W_2 的参数量均为 768 * (768 * 4)，总参数量为 768 * 768 * 4 * 2。

        总参数量 = 词向量参数 + 12 (层数) * (Multi-head + 前馈网络) = 110M

5. ALBERT

    - Factorized Embedding Parameterization

        在 BERT 中，模型直接将词表对应到 word embedding 中，embedding 的维度大小和隐藏层 H 的维度大小相等。这是没有必要的，因为当维度大小 $d_H$ 增加时，word embedding 维度的增加没有意义。因此引入多一层转换矩阵 E，让词表 V 先通过转换矩阵，再转换为隐藏层的维度大小。这样可以明显减小参数量，由之前的 $(d_V * d_H)$ 减少为 $(d_V * d_E + d_E * d_H)$。
    
    - Cross-Layer Parameter Sharing

        BERT 框架中的参数主要包括 Attention 层的参数和 Feed Forward 网络的参数，ALBERT 将这些参数都共享，大大减小了参数量，为了弥补性能的损失，ALBERT 加大了隐藏层的维度大小，由“窄而深”变成“宽而浅”。

    - Sentence Order Prediction 

        针对 BERT 的第二个训练任务，即判断 A 是否是 B 的下一句话，过于简单的问题，ALBERT 增加了预训练的难度，即将负样本换成了两个句子的逆序排列。

        > **[NSP 任务]** 正样本：同一个文档的两个连续句子；负样本：两个连续句子交换顺序
        > 
        > **[SOP 任务]** 正样本：同一个文档的两个连续句子；负样本：不同文档的句子


6. XLNet

    由于 BERT 在预训练过程中需要加入 [mask]，而在下游任务及预测过程中都没有这样的标记，因此会造成性能损失。XLNet 则通过自回归语言模型的思想来解决，即从左到右依次生成。为了保持模型仍然是双向的，能够同时从前文和后文获取信息，XLNet 引入了 Attention Mask 机制。

    假设模型在预训练过程中需要预测第 $k$ 个词，那么首先先将序列随机打乱，再取前 $k-1$ 个词进行预测，这样既可以读到前后文的信息，又可以省去 [mask] 标记。

    这样的预训练模式天然符合下游任务序列生成的任务，因此可以预计 XLNet 在文本摘要，机器翻译，信息检索等领域具有优势。


7. TinyBERT

    由于 BERT 模型过于庞大，很难实际应用落地。因此提出了一种蒸馏 BERT 的方法 TinyBERT，它的大小不到 BERT 的 1/7，但速度提高了 9 倍。

    知识蒸馏的基本思想是使用一个大的训练好的模型来知道小模型更好的训练。TinyBERT 的基本思想是减少 Transformer 的层数以及降低 hidden_size 的大小。模型结构如下：

    ![TinyBERT](imgs/TinyBERT.jpg)

    TinyBERT 的 loss 分为三部分：

    - Embedding Layer Distillation

        TinyBERT 的 embedding 大小比教师模型更小，因此需要通过一个维度变换矩阵来把学生模型的 embedding 映射到教师模型所在空间，再通过 MSE 来计算 loss：
        $$
            \mathcal{L}_{embd}= \text{MSE}(E^SW_e, E^T)
        $$

    - Transformer Layer Distillation

        TinyBERT 的知识蒸馏采取每隔 k 层蒸馏的方式。设 Teacher BERT 有 12 层，TinyBERT 有 4 层，则学生模型每隔 3 层就与教师模型计算一次 loss，其中，loss 又分为 Attention Loss 和 Hidden Loss：

        $$
            \mathcal{L}_{attn} = \frac{1}{h}\sum_{i=1}^h \text{MSE}(A_i^S, A_i^T)
        $$
        其中，h 为 Attention 头数，$A_i\in \{A_q,A_k,A_v\}$。

        $$
            \mathcal{L}_{hidn} = \text{MSE}(H^SW_h, H^T)
        $$

    - Prediction Layer Distillation

        在预测层的 loss 计算取决于不同的具体任务，但都需要结合教师模型和学生模型的 loss。


8. RoBERTa

    - 去除 NSP 任务
    - 动态掩码。RoBERTa的做法是将训练数据复制多份，每份采用不同的随机挑选 token 进行掩码。这样有助于模型适应不同的掩码策略，学习不同的语言表征。
    - 文本编码。使用了更大的词汇表来训练。
    - 可以看作是一个“调过参的 BERT 版本”



