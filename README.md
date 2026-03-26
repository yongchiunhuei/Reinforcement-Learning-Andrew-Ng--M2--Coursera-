# 🎓 Machine Learning Specialization: Module 3 (Deep Dive)
## Unsupervised Learning, Recommenders, & Reinforcement Learning

This module focuses on models that find hidden structures, predict preferences, and learn optimal strategies through environmental interaction.

---

### 1. Clustering (聚类 - Clustering)
Clustering is used when you have data but no "ground truth" labels.

#### K-means Deep Dive
* **Optimization Objective:** Minimize the **Distortion Cost Function** ($J$).
  $$J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_K) = \frac{1}{m} \sum_{i=1}^{m} ||x^{(i)} - \mu_{c^{(i)}}||^2$$
* **Random Initialization:** To avoid local optima, run K-means 50–1000 times with different random initializations and pick the one with the lowest cost $J$.
* **The Elbow Method:** Plotting $K$ vs. Cost. If the curve is smooth with no clear "elbow," choose $K$ based on the **downstream purpose** (e.g., if you need 3 sizes for t-shirts, $K=3$).



---

### 2. Anomaly Detection (异常检测 - Anomaly Detection)
Unlike supervised learning, anomaly detection handles datasets where the "positive" class (anomalies) is extremely small ($n < 50$).

#### Gaussian Distribution Method
We assume each feature $x_j$ follows a Gaussian distribution with mean $\mu_j$ and variance $\sigma_j^2$.
1. **Estimate Parameters:** Calculate $\mu$ and $\sigma$ for each feature.
2. **Compute Probability:** $p(x) = p(x_1;\mu_1, \sigma_1^2) \times p(x_2;\mu_2, \sigma_2^2) \times ...$
3. **Threshold:** If $p(x) < \epsilon$, flag as anomaly.

**Anomaly Detection vs. Supervised Learning:**
* Use **Anomaly Detection** when you have a very small number of positive examples (0-20) and many different "types" of anomalies that haven't been seen yet.
* Use **Supervised Learning** when you have a large number of positive and negative examples, and future anomalies are likely to look like the ones in your training set.

---

### 3. Recommender Systems (推荐系统 - Recommender Systems)
This is the bridge between data science and user experience.

#### Collaborative Filtering (协同过滤)
* **The Concept:** If User A and User B both highly rated "The Matrix," and User A likes "Inception," User B likely will too.
* **Vector Embeddings:** The model learns a feature vector $x^{(i)}$ for each movie and a parameter vector $w^{(u)}$ for each user.
* **Mean Normalization:** Always subtract the average rating of a movie before training. This ensures that a user who hasn't rated anything yet is predicted to give the "average" rating rather than 0 stars.

#### Content-Based Filtering
* Use a **Deep Learning** approach (Neural Networks).
* Input user features (age, gender, history) into a "User Network."
* Input movie features (genre, year, actors) into a "Movie Network."
* The output is a dot product of the two network vectors ($v_u \cdot v_m$) to predict the rating.



---

### 4. Reinforcement Learning (强化学习 - Reinforcement Learning)
The most "sentient" feeling part of AI—learning from trial and error.

#### The Markov Decision Process (MDP)
* **State ($s$):** The environment's current configuration.
* **Action ($a$):** The choice made by the agent.
* **Reward ($R$):** The score received (can be delayed).
* **Policy ($\pi$):** The "brain" or mapping from $s$ to $a$. Our goal is to find the optimal policy $\pi^*$.

#### State-Action Value Function ($Q(s, a)$)
$Q(s, a)$ is the return if you start in state $s$, take action $a$, and behave optimally thereafter.
* **Exploration (探索):** Taking a random action to discover new rewards.
* **Exploitation (利用):** Taking the best-known action to get a guaranteed reward.
* **$\epsilon$-greedy Policy:** A simple way to balance both. Pick a random action with probability $\epsilon$ (e.g., 0.1), otherwise pick the best action.

---

### 🛠️ Advanced Tools Checklist
- [ ] **Scikit-Learn:** For K-means and basic Gaussian modeling.
- [ ] **TensorFlow/Keras:** For building Recommender "Towers."
- [ ] **Gymnasium (OpenAI Gym):** The standard toolkit for Reinforcement Learning environments.

> **Ninja Note:** In Recommender Systems, beware of the "Filter Bubble." If you only recommend what users already like, they never discover anything new. Inject a little "Exploration" into your recommendations!
