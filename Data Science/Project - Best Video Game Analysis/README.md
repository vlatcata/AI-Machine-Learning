This projects aims to scientifically calcuclate the best performing video games (on Steam) by using a custom BGI (Best Game Index).

**Context:**  
The video game industry is huge, with thousands of games released every year. Only a small part of them manage to become popular and keep players engaged. Finding out what makes these games successful can help both developers and players.  

**Aims:**  
The goal of this project is to build a **Best Game Index (BGI)** that combines several factors into one score, showing which games stand out.  

**Methods:**  
I first built a custom scraper to collect data from the **Steam API** and **SteamSpy API**, but due to strict request limits it was too slow to use directly. Instead, I used a ready dataset with the same information. From it, I created new features like a *review score* (that gives more weight to positive reviews) and a *playtime score*. All features were normalized to be on the same scale. A weighted formula was then used to calculate the BGI. To test if my weights were reliable, I added random noise and checked if the top games stayed the same.  

**Results:**  
Most games in the dataset perform poorly, while only a few dominate in ownership, playtime, and community engagement. The BGI proved stable: even after adding noise to the weights, the top-10 list of games stayed about 90% the same. This shows the index is not too dependent on exact weight choices and works well to highlight the most successful games.  
