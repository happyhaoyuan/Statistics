# Statistics

## Random Variable

### Discrete Distribution

|              Distribution              |           **Poisson**: <br>$X\sim Pois(\lambda)$           |    Bernoulli:<br> $X\sim Ber(p)$    |          Binomial Distribution:<br>$X\sim B(n,p)$           | Geometric Distribution:<br>$X\sim Geo(p)$ |         Negative binomial:<br>$X\sim NB(k,p)$         |    Multinomial:<br>$X=(X_1,...,X_k)\sim M(n,p_1,...,p_k)$    |
| :------------------------------------: | :--------------------------------------------------------: | :---------------------------------: | :---------------------------------------------------------: | :---------------------------------------: | :---------------------------------------------------: | :----------------------------------------------------------: |
|             **Parameters**             |                        $\lambda >0$                        |            $0\le p\le1$             |                     $n\in N,0\le p\le1$                     |               $p \in (0,1)$               |                   $k>0,p\in (0,1)$                    |                $n>0,\sum\limits_{i=1}^kp_i=1$                |
|              **Support**               |                       $x \in N^{+}$                        |           $x \in \{0,1\}$           |                    $X \in \{0,1,...,n\}$                    |            $X \in \{1,2,...\}$            |                 $X\in \{k,k+1,...\}$                  | $X_i \in \{0,...,n\},i=1,...,k$<br>$\sum\limits_{i=1}^kX_i=n$ |
|                **PMF**                 | $P(X=k)=\frac{\lambda^ke^{-\lambda}}{k!}$<br>$k=0, 1, ...$ | $P(X=k)=p^k(1-p)^{1-k}$<br>$k=0, 1$ |    $P(X=k)={n\choose k}p^k(1-p)^{n-k}$<br>$ 0\le k\le n$    |     $P(X=k)=p(1-p)^{k-1}$<br>$k\ge 1$     | $P(X=n)={n-1\choose k-1}p^{k}(1-p)^{n-k}$<br>$n\ge k$ | $P(X_1=n_1,...,X_k=n_k)=$<br>$\frac{n!}{\prod\limits_{i=1}^k n_i!}\prod\limits_{i=1}^{k}p_i^{n_i}$ |
|                **CDF**                 |                             -                              |                  -                  |                              -                              |            $P(X<=k)=1-(1-p)^k$            |                           -                           |                              -                               |
|                **Mean**                |                       $E(X)=\lambda$                       |              $E(X)=p$               |                          $E(X)=np$                          |            $E(X)=\frac{1}{p}$             |                  $E(X)=\frac{k}{p}$                   |                        $E(X_i)=np_i$                         |
|              **Variance**              |                      $Var(X)=\lambda$                      |           $Var(X)=p(1-p)$           |                      $Var(X)=np(1-p)$                       |         $Var(X)=\frac{1-p}{p^2}$          |              $Var(X)=\frac{k(1-p)}{p^2}$              |                    $Var(X_i)=np_i(1-p_i)$                    |
|                **MLE**                 |                  $\hat{\lambda}= \bar{X}$                  |          $\hat{p}=\bar{X}$          |                              -                              |        $\hat{p}=\frac{1}{\bar{X}}$        |              $\hat{p}=\frac{k}{\bar{X}}$              |                $\hat{p}=\frac{X_i}{\bar{n}}$                 |
| **Fisher Info**<br>(**Single Sample**) |               $I(\lambda)=\frac{1}{\lambda}$               |       $I(p)=\frac{1}{p(1-p)}$       |                   $I(p)=\frac{n}{p(1-p)}$                   |         $I(p)=\frac{1}{p^2(1-p)}$         |               $I(p)=\frac{k}{p^2(1-p)}$               |                              -                               |
|               **Others**               |                                                            |                                     | $X=\sum\limits_{i=1}^n Y_i,Y_i \overset{i.i.d}{\sim}Ber(p)$ |   Memoryless:<br>$P(X>m+n|X>n)=P(X>m)$    |                   $NB(1,p)=Geo(p)$                    |                                                              |

### Continues Distribution

|            Distribution            |                 Uniform: <br>$X\sim U(a,b)$                  |            Exponential:<br> $X\sim Exp(\lambda)$             |       Normal Distribution:<br>$X\sim N(\mu,\sigma^2)$        |           Gamma:<br>$X\sim \Gamma(\alpha，\beta)$            |             Beta:<br>$X\sim Beta(\alpha,\beta)$              |           Chi-squared:<br>$X\sim \mathcal{X}^2_n$            |                Student's *t*:<br>$X\sim t_n$                 |
| :--------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|           **Parameters**           |                     $-\infty<a<b<\infty$                     |                         $\lambda>0$                          |                    $\mu\in R, \sigma^2>0$                    |                      $\alpha>0,\beta>0$                      |                      $\alpha>0,\beta>0$                      |                          $n\in N^+$                          |                            $n>0$                             |
|            **Support**             |                        $x \in [a,b]$                         |                           $x\ge 0$                           |                          $x \in R$                           |                            $X>0$                             |                        $X \in [0,1]$                         |                      $x\in [0,\infty)$                       |                   $x\in (-\infty,\infty)$                    |
|              **PDF**               | $f(x)=  \begin{cases}     \frac{1}{b-a}& x\in [a,b]\\     0              & otherwise \end{cases}$ |                $f(x)=\lambda e^{-\lambda x}$                 | $f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ |  $f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$  | $f(x)=\frac{1}{2^{\frac{n}{2}}\Gamma(\frac{n}{2})}x^{\frac{n}{2}-1}e^{\frac{x}{2}}$ |                              -                               |
|              **CDF**               | $f(x)=  \begin{cases} 0& x<a\\ \frac{x-a}{b-a}& x\in [a,b]\\ 1              & x>b \end{cases}$ |                   $F(x)=1-e^{-\lambda x}$                    |                              -                               |                              -                               |                              -                               |                              -                               |                              -                               |
|              **Mean**              |                     $E(X)=\frac{a+b}{2}$                     |                   $E(X)=\frac{1}{\lambda}$                   |                          $E(X)=\mu$                          |                 $E(X)=\frac{\alpha}{\beta}$                  |              $E(X)=\frac{\alpha}{\alpha+\beta}$              |                           $E(X)=n$                           |                           $E(X)=0$                           |
|            **Variance**            |                 $Var(X)=\frac{(b-a)^2}{12}$                  |                 $Var(X)=\frac{1}{\lambda^2}$                 |                      $Var(X)=\sigma^2$                       |               $Var(X)=\frac{\alpha}{\beta^2}$                | $Var(X)=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |                         $Var(X)=2n$                          |                  $Var(X)=\frac{n}{n-2},n>2$                  |
|              **MLE**               | $  \begin{cases}     \hat{b}=X_{(n)}& \text{known $a$},\text{unknown $b$}\\     \hat{a}=X_{(1)}              & \text{unknown $a$},\text{known $b$} \end{cases}$ |              $\hat{\lambda}=\frac{1}{\bar{X}}$               | $logL(\mu,\sigma^2)=-\frac{n}{2}ln(2\pi)-\frac{n}{2}ln\sigma^2-\frac{1}{2\sigma^2}\sum\limits_{i=1}^{n}(x_i-\mu)^2$<br>$  \begin{cases}     \hat{\mu}=\bar{X}& \text{unknown $\mu$},\text{known $\sigma^2$}\\     \hat{\sigma}^2=\frac{\sum\limits_{i=1}^{n}{(X_i-\mu)^2}}{n}& \text{known $\mu$},\text{unknown $\sigma^2$}\\ \hat{\mu}=\bar{X}, \hat{\sigma}^2=\frac{\sum\limits_{i=1}^{n}(X_i-\bar{X})^2}{n} & \text{unknown $\mu$},\text{unknown $\sigma^2$} \end{cases}$ |                              -                               |                              -                               |                              -                               |                              -                               |
| **Fisher Info<br>(Single Sample)** |                              -                               |               $I(\lambda)=\frac{1}{\lambda^2}$               | $  \begin{cases}     I(\mu)=\frac{1}{\sigma^2}& \text{unknown $\mu$},\text{known $\sigma^2$}\\     I(\sigma^2)=\frac{1}{2\sigma^4}& \text{known $\mu$},\text{unknown $\sigma^2$}\end{cases}$ |                              -                               |                              -                               |                              -                               |                              -                               |
|             **Others**             |                                                              | - Memoryless: <br>$P(X>s+t|X>t)=P(X>s)$<br>- Order Statistics:<br>$X_{(1)}\sim Exp(n\lambda)$<br>- Scaling:<br>$aX\sim Exp(\lambda/a),a>0$<br>- $Exp(\frac{1}{2})=\mathcal{X}^2_2$ | - $X\sim N(\mu,\sigma^2),aX+b\sim N(a\mu+b,a^2\sigma^2)$<br>- $X\sim N(\mu_1,\sigma_1^2),Y\sim N(\mu_2,\sigma_2^2),$<br>$aX+bY\sim N(a\mu_1+b\mu_2,a^2\sigma_1^2+b^2\sigma_2^2+2ab\sigma_1\sigma_2\rho)$ | - $X_i \overset{indep}{\sim}\Gamma(\alpha_i,\beta),$<br>$\sum X_i \sim \Gamma(\sum\alpha_i,\beta)$<br>- $\Gamma(1,\lambda)=Exp(\lambda)$<br>- $\Gamma(\frac{n}{2},\frac{1}{2})=\mathcal{X}^2_n$ |                                                              | - $Z_i\overset{i.i.d}{\sim}N(0,1),$<br>$V=\sum Z_i\sim \mathcal{X}^2_n$<br>- $\mathcal{X}^2_n=\Gamma(\frac{n}{2},\frac{1}{2})$ | $Z\sim N(0,1)$ independent with $V~\sim \mathcal{X}^2_n,$$T=\frac{Z}{\sqrt{\frac{V}{n}}}\sim t_n$ |



## Hypothesis Testing

### Concepts

+ **Null Hypothesis**: $H_{0}$ vs **Alternative Hypothesis **: $H_{1}$

- **Test Statistic**: $T$

  + 是随机变量，原假设下分布 $T|H_0 \sim F$

  + 给定$\alpha$，可根据原假设求$RR$

    $RR=\begin{cases} \{|T|>F_{1-\frac{\alpha}{2}}\} & H_0:\theta=\theta_0 \\ {\{T>F_{1-\alpha}}\} & H_0:\theta\le\theta_0 \\ \{T<-F_{\alpha}\} & H_0:\theta\ge\theta_0\end{cases}$

  + 是样本的函数$T(X)$

  + reject $H_0$ if $T(X) \in RR$

- **Rejection Region**: $RR$

- **Type I Error** (**Significance Level**): $\alpha=P(T\in RR|H_0)$

  rejecting $H_0$ when $H_0$ is true

- **Type II Error**: $\beta=P(T\notin RR|H_1)$

  accepting $H_0$ when $H_0$ is false

  for **specific value** of the alternative hypothesis: $\beta(\theta_1)=P(T\notin RR|\theta=\theta_1\in \Theta_1)$

- **Statistical Power**: $1-\beta$

- **p-value**: $P(T$ *is extreme than* $T_{obs}|H_0)$

  1. 找分布和统计量的样本观测值

      $T|\theta=\theta_0\sim F,T_{0}=T(X)$

  2. 求$p$-$value$

     $p$-$value=\begin{cases} P(|T|>T_0|T\sim F) & H_0:\theta=\theta_0 \\ P(T>T_0|T\sim F) & H_0:\theta\le\theta_0 \\ P(T<T_0|T\sim F) & H_0:\theta\ge\theta_0 \end{cases}$

  3. 给定 $\alpha$

     $\begin{cases} \text{reject $H_0$} & p\text{-}value\le\alpha \\ \text{accept $H_0$} & p\text{-}value>\alpha \end{cases}$

### Likelihood Ratio Tests(Simple Hypothesis)

$H_0: \theta=\theta_0$ vs $H_1: \theta=\theta_1$

1. 写出 Likelihood function: $L(\theta)=\prod\limits_{i=1}^{n}f(x_i|\theta)$

2. 求 Likelihood ratio: $\Lambda(X)=\frac{L(\theta_0)}{L(\theta_1)}$

3. threshold $C$，得到拒绝域$RR=\{\Lambda(X)\le C\}$

   若$C$未知，给定$\alpha$，令$\alpha=P(\Lambda(X)\le C|X\sim F_{\theta_0})\overset{通过\Lambda反解出X}{=}P(X\le\Lambda^{-1}(C)|X\sim F_{\theta_0})$可求得C

4. 做决策

   $\begin{cases} \text{reject $H_0$} & \Lambda\le C \\ \text{accept $H_0$} & \Lambda> C \end{cases}$

### Generalized Likelihood Ratio Tests

$H_0: \theta \in \Theta_0$ vs $H_1: \theta \in \Theta_1$

$\Theta=\Theta_0\cup\Theta_1, d=dim(\Theta), d_0=dim(\Theta_0)$ 

1. 写出 Likelihood function: $L(\theta)=\prod\limits_{i=1}^{n}f(x_i|\theta)$

2. 求$\Theta_0$和$\Theta$下的MLE: 

   $\hat{\theta_0}=\underset{\theta\in \Theta_0}{argMax}L(\theta)$

   $\hat{\theta}=\underset{\theta\in \Theta}{argMax}L(\theta)$

3. 求 Likelihood ratio: $\Lambda=\frac{L(\hat{\theta_0})}{L(\hat\theta)}$

4. 求分布：$-2log\Lambda\sim\mathcal{X}_{d-d_{0}}^2$

5. 给定$\alpha$

   拒绝域$RR=\{-2log\Lambda>F_{\mathcal{X}_{d-d_{0}}^2}(1-\alpha)\}$

   $\begin{cases} \text{reject $H_0$} & -2log\Lambda>F_{\mathcal{X}_{d-d_{0}}^2}(1-\alpha) \\ \text{accept $H_0$} & -2log\Lambda\le F_{\mathcal{X}_{d-d_{0}}^2}(1-\alpha) \end{cases}$

   或求$p$-$value=1-F_{\mathcal{X}_{d-d_{0}}^2}^{-1}(-2log\Lambda)$

   $\begin{cases} \text{reject $H_0$} & p\text{-}value\le\alpha \\ \text{accept $H_0$} & p\text{-}value>\alpha \end{cases}$

### Pearson's $\mathcal{X}^2$ Test

- statistics $\mathcal{X}^2=\sum\limits$
- Consistent Test
- Independent Test
- 



### Normal Distribution

$X\sim N(\mu,\sigma^2)$

- $\mu$
- $\sigma^2$

1. 



样本均值sample mean: $\bar{X}=\frac{\sum\limits_{i=1}^{n}X_i}{n}$

样本方差sample variance: $S^2=\frac{\sum\limits_{i=1}^n(X_i-\bar{X})^2}{n-1}=\frac{\sum\limits_{i=1}^{n}X_{i}^2-n\bar{X}^2}{n-1}$

