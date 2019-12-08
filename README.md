# Statistics

## Distribution Summary

### Discrete Distribution

1. **Poisson Distribution**

   $X\sim Pois(\lambda), \lambda >0$

   - Support: $x \in N^{+}$
   - PMF: $P(X=k)=\frac{\lambda^ke^{-\lambda}}{k!}, k=0, 1, ...$
   - Mean: $E(X)=\lambda$
   - Variance: $Var(X)=\lambda$
   - MLE: $\hat{\lambda}= \bar{X}$
   - Fisher Information(Single Sample): $I(\lambda)=\frac{1}{\lambda}$

2. **Bernoulli Distribution**

   $X\sim Ber(p), 0\le p\le1$

   - Support: $x \in \{0,1\}$
   - PMF: $P(X=k)=p^k(1-p)^{1-k}, k=0, 1$
   - Mean: $E(X)=p$
   - Variance: $Var(X)=p(1-p)$
   - MLE: $\hat{p}=\bar{X}$
   - Fisher Information(Single Sample): $I(p)=\frac{1}{p(1-p)}$

3. **Binomial Distribution**

   $X\sim B(n,p),n\in N, 0\le p\le1$

   $X=\sum\limits_{i=1}^n Y_i,Y_i \overset{i.i.d}{\sim}Ber(p)$

   - Support: $X \in \{0,1,...,n\}$
   - PMF: $P(X=k)={n\choose k}p^k(1-p)^{n-k}, 0\le k\le n$
   - Mean: $E(X)=np$
   - Variance: $Var(X)=np(1-p)$
   - Fisher Information(Single Sample): $I(p)=\frac{n}{p(1-p)}$

4. **Geometric Distribution**

   $X\sim Geo(p),p \in (0,1)$

   - Support: $X \in \{1,2,...\}$
   - PMF: $P(X=k)=p(1-p)^{k-1}, k\ge 1$
   - CDF:$P(X<=k)=1-(1-p)^k$
   - Mean: $E(X)=\frac{1}{p}$
   - Variance: $Var(X)=\frac{1-p}{p^2}$
   - MLE: $\hat{p}=\frac{1}{\bar{X}}$
   - Fisher Information(Single Sample): $I(p)=\frac{1}{p^2}+\frac{1}{p(1-p)}$

5. 

### Continues Distribution

1. **Normal Distribution**

   $X\sim N(\mu,\sigma^2), \mu\in R, \sigma^2>0$

   - Support: $x \in R$

   - PDF: $f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

   - Mean: $E(X)=\mu$

   - Variance: $Var(X)=\sigma^2$

   - MLE

     $logL(\mu,\sigma^2)=-\frac{n}{2}ln(2\pi)-\frac{n}{2}ln\sigma^2-\frac{1}{2\sigma^2}\sum\limits_{i=1}^{n}(x_i-\mu)^2$

     - unknown $\mu,$known $\sigma^2$

       $\hat{\mu}=\bar{X}$

     - known $\mu,$unknown $\sigma^2$

       $\hat{\sigma}^2=\frac{\sum\limits_{i=1}^{n}{(X_i-\mu)^2}}{n}$

     - unknown $\mu,$unknown $\sigma^2$

       $\hat{\mu}=\bar{X}, \hat{\sigma}^2=\frac{\sum\limits_{i=1}^{n}(X_i-\bar{X})^2}{n}$

   - Fisher Information(Single Sample)

     - unknown $\mu,$known $\sigma^2$

       $I(\mu)=\frac{1}{\sigma^2}$

     - known $\mu,$unknown $\sigma^2$

       $I(\sigma^2)=\frac{1}{2\sigma^4}$

2. **Exponential distribution**

   $X\sim Exp(\lambda),\lambda>0$

   - Support: $x\ge 0$
   - PDF: $f(x)=\lambda e^{-\lambda x}$
   - CDF:$F(x)=1-e^{-\lambda x}$
   - Mean: $E(X)=\frac{1}{\lambda}$
   - Variance: $Var(X)=\frac{1}{\lambda^2}$
   - MLE: $\hat{\lambda}=\frac{1}{\bar{X}}$
   - Fisher Information(Single Sample):$I(\lambda)=\frac{1}{\lambda^2}$



## Hypothesis Testing

1. Concepts

+ **Null Hypothesis**: $H_{0}$

- **Alternative Hypothesis **: $H_{1}$

- **Rejection Region**: $RR$

- **Test Statistic**: $T$

  + combine the random variables/function of the samples
  + consider as an estimator, $T|H_0 \sim F$
  + reject $H_0$ if $T \in RR$

- **Type I Error** (**Significance Level**): $\alpha=P(T\in RR|H_0)$

  rejecting $H_0$ when $H_0$ is true

- **Type II Error**: $\beta=P(T\notin RR|H_1)$

  accepting $H_0$ when $H_0$ is false

  for specific value of the alternative hypothesis: $\beta(\theta_1)=P(T\notin RR|\theta=\theta_1\in \Theta_1)$

- **Statistical Power**: $1-\beta$

- **p-value**: $P(T$ *is extreme than* $T_{obs}|H_0)$

  for significant level $\alpha$

  - reject $H_0$ if  $p$-$value\le\alpha$
  - accept $H_0$ if  $p$-$value>\alpha$

2. **Likelihood Ratio Tests(Simple Hypothesis)**

   $H_0: \theta=\theta_0$ vs $H_1: \theta=\theta_1$

   - Likelihood function: $L(\theta)=\prod\limits_{i=1}^{n}f(x_i|\theta)$

   - Likelihood ratio: $LR=\frac{L(\theta_0)}{L(\theta_1)}$
   - reject $H_0$ when $LR<C,$ compute $C$ by solve $\alpha=P(LR<C|H_0)$

3. **Generalized Likelihood Ratio Tests**

   $H_0: \theta \in \Theta_0$ vs $H_1: \theta \in \Theta_1$

   $\Theta=\Theta_0\cup\Theta_1, d=dim(\Theta), d_0=dim(\Theta_0)$ 

   Likelihood function: $L(\theta)=\prod\limits_{i=1}^{n}f(x_i|\theta)$

   Likelihood ratio: $\Lambda=\frac{\underset{\theta\in\Theta_0}{Max}L(\theta)}{\underset{\theta\in\Theta}{Max}L(\theta)}$

   - $\hat{\theta}=\underset{\theta}{argMax}$ is MLE of $\theta,$ depends on space $\Theta_0$ or $\Theta$

   - $-2log\Lambda\sim\mathcal{X}_{d-d_{0}}^2$

   - for significant level $\alpha$

     + $RR=\{-2log\Lambda>F_{\mathcal{X}_{d-d_{0}}^2}^{-1}(1-\alpha)\}, F$ is CDF

     + reject $H_0$ if $-2log\Lambda>F_{\mathcal{X}_{d-d_{0}}^2}^{-1}(1-\alpha)$
     + reject $H_0$ if $p$-$value=1-F_{\mathcal{X}_{d-d_{0}}^2}^{-1}(-2log\Lambda)\le\alpha$

4. **Normal Distribution**

   $X\sim N(\mu,\sigma^2)$

   - $\mu$
   - $\sigma^2$

5. **Pearson's** $\mathcal{X}^2$ **Test**

   - statistics $\mathcal{X}^2=\sum\limits$
   - Consistent Test
   - Independent Test
   - 

6. 



样本均值sample mean: $\bar{X}=\frac{\sum\limits_{i=1}^{n}X_i}{n}$

样本方差sample variance: $S^2=\frac{\sum\limits_{i=1}^n(X_i-\bar{X})^2}{n-1}=\frac{\sum\limits_{i=1}^{n}X_{i}^2-n\bar{X}^2}{n-1}$

