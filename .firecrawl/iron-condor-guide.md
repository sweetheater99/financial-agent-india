# What Is Iron Condor Strategy ?Examples, Meaning & How It Works

![What Is Iron Condor Strategy ?Examples, Meaning & How It Works](https://choiceindia.com/blogs/content/images/2025/11/Iron-Condor-Strategy.png)

- Published Date: November 26, 2025
- Updated Date: November 26, 2025
- By Team Choice

##### Table of Contents

- What Is the Iron Condor Strategy?
- How to Set Up an Iron Condor?
- How Does an Iron Condor Work?
- Iron Condor Strategy Example
- Important for Indian Traders:
- When Does Loss Begin?
- Iron Condor Profit and Loss
- Breakeven Points
- What Determines Profitability?
- Conclusion
- FAQs

For many Indian [future and options](https://choiceindia.com/future-and-option) traders, the goal is simple: to earn a steady income with limited risk. The Iron Condor Option strategy is one of the most popular ways to achieve this, particularly when the market is expected to stay within a range.

This blog explains what is an iron condor strategy, how to set it up, how it works, iron condor strategy example, profit/loss, and margin requirements.

## What Is the Iron Condor Strategy?

The Iron Condor strategy is an options trading technique that involves selling two credit spreads: one call spread and one put spread, to profit from a market that trades within a range. It is primarily used on index options, such as NIFTY and BANK NIFTY in India.

Think of the Iron Condor as a strategy that bets the market not making a big move. Instead of predicting whether the market will go up or down, you’re predicting it will stay calm and trade in a range.

**There are two types:**

- **Short Iron Condor:** The widely used version where traders collect premium and benefit from time decay.
- **Long Iron Condor:** A debit version designed to profit from volatility expansion.

Most traders in India use the **short iron condor** since it offers a higher probability of profit.

The Iron Condor is considered a defined-risk strategy because the hedge options you purchase cap your maximum loss. In the Indian market, this works to your benefit; brokers block far less margin for an iron condor than for naked option selling, making it a much more capital-efficient choice for traders who actively trade index options.

## How to Set Up an Iron Condor?

Setting up an Iron Condor may look complicated at first, but it becomes very straightforward once you understand the four legs involved. The goal is to establish a range around the current market price within which you expect the index to stay until expiry.

Here’s a step-by-step explanation:

### 1\. Identify the Expected Range:

Before placing any trades, decide the price range you believe NIFTY or BANK NIFTY will stay within until expiry. You can determine this range using Support and resistance levels, Options data (OI and PCR), India VIX, Market trend or lack of trend, and Expected news or events. Once you have the range, you can choose your option strikes.

Note:

**The Golden Rule: Spread Width (Risk vs. Reward)**

The distance between your sold strike and your bought strike (the spread width) determines your maximum risk and the margin blocked.

- **Narrow Spreads:**(e.g., 50 points wide on NIFTY) offer a higher premium relative to their spread width, offering a better risk-to-reward ratio. However, they also narrow your safety range.
- **Wide Spreads:** (e.g., 100 points wide on NIFTY) give you a lower risk-to-reward ratio but offer a wider, safer breakeven range and a higher probability of profit.

**Best Practice:** Many Indian traders use 100-point spreads for NIFTY and 200-point spreads for BANK NIFTY to balance premium collection and risk containment.

### 2\. Sell an Out-of-the-Money (OTM) Call Option:

This strike should be above the current market price and beyond the level you think the market will stay below. For example, if NIFTY is at 22,000, you may sell a 22,200 CE. This is one of the premium-earning legs.

### 3\. Buy a Further OTM Call Option (Protection):

To limit your risk, you buy another call option with a higher strike. For example, buy 22,300 CE. This creates a bear call spread, capping your loss if the market unexpectedly moves upward.

### 4\. Sell an Out-of-the-Money Put Option:

Next, choose a strike below the current market price, where you believe the index is unlikely to fall. For example, sell 21,800 PE. This is the second premium-earning leg.

### 5\. Buy a Further OTM Put Option (Protection):

To hedge the downside, buy another put option at a lower strike price. For example, buy 21,700 PE. This completes the bull put spread.

### 6\. Confirm Net Credit:

Since you’re selling two options that are more expensive than the ones you’re buying, the strategy gives you a net credit (premium received upfront). This credit is your maximum possible profit.

### 7\. Check Margin Requirements:

Because this is a defined-risk strategy, Indian brokers block margin only for the maximum possible loss, not for unlimited exposure. This makes the Iron Condor much more capital-efficient than naked option selling.

Margins are usually lower when:

- Strikes are further apart
- Market volatility is moderate
- Risk is clearly defined by the long options

### 8\. Place All Four Legs Together:

Most Indian brokers allow you to place multi-leg strategies in one go using:

- Strategy Builder
- Basket Orders
- Options Strategy Panels

This helps avoid execution issues and keeps slippage low.

### 9\. Monitor and Adjust if Needed:

Your goal is for the index to expire between your two short strikes, so that all options expire worthless and you keep the full premium. If the index starts moving toward one side, you can adjust the untested side or exit the position early to contain losses.

## How Does an Iron Condor Work?

Here’s how the Iron Condor works:

### 1\. You Sell Two Out-of-the-Money Options to Earn Premium:

The strategy begins by selling one call option above the current market price and one put option below the current price. These two sold options bring in premium upfront. Your goal is for both of them to expire worthless so that you can keep the entire premium.

### 2\. You Buy Two Further Out-of-the-Money Options for Safety:

To protect yourself from large market movements, you also buy a higher strike call option and a lower strike put option. These protective options ensure your maximum loss remains capped, making the strategy safer and margin-efficient in the Indian market.

### 3\. You Receive a Net Credit at the Start:

Because the options you sell are more expensive than the ones you buy, the entire position results in a net credit. This net credit is your maximum profit. You earn this profit if the underlying index stays within the range of your sold strikes until expiry.

### 4\. Time Decay and Volatility Help You:

The Iron Condor profits from two core Greeks:

- **Time Decay (θ):** As expiry approaches, option premiums naturally decrease. This reduction in premium benefits the sold options, which is where your profit primarily comes from. This is why many traders prefer using Iron Condors on weekly expiries in NIFTY and BANK NIFTY.
- **Volatility Drop (** _**v**_ **):** The Iron Condor is a Vega-negative strategy. A decrease in implied volatility (a drop in India VIX) causes the value of your short options to decline faster than your long options, accelerating your profit. This is why trading the Condor when VIX is high often yields better results.

### 5\. You Profit When the Market Stays in a Range:

The Iron Condor works best when the market remains between your short call strike and your short put strike. If the index stays within this range till expiry, all options expire worthless, and you retain the entire premium.

### 6\. Loss Starts Only When Price Moves Beyond the Breakeven Levels:

If the market breaks out of the range, the position begins to lose value. Losses start beyond the breakeven points and increase gradually. However, the maximum loss is capped because the protective options you bought limit the risk.

### 7\. Your Risk Is Fixed and Known in Advance:

The difference between the spread width and the net premium received determines your maximum loss. This makes the Iron Condor a defined-risk strategy with predictable outcomes, which is why margin requirements are lower in India compared to naked selling.

### 8\. The Strategy Performs Well in Stable or Sideways Markets:

The Iron Condor is most effective when the market is not trending strongly. It performs well in weeks with low news flow, stable volatility, and a clear range. Many traders use it to generate a consistent income during calm market conditions.

## Iron Condor Strategy Example

**Assume NIFTY =** 22,000.

You expect NIFTY to stay between 21,800 and 22,200 by expiry.

Create this short iron condor:

- Sell 22,200 CE
- Buy 22,300 CE
- Sell 21,800 PE
- Buy 21,700 PE

**Net premium received:** ₹40

**Spread width:** 100 points

### Breakeven Points

- **Upper Breakeven:** 22,200 + 40 = 22,240
- **Lower Breakeven:** 21,800 – 40 = 21,760

### Maximum Profit

= Net premium received

= ₹40 per unit

## Important for Indian Traders:

Options in India are traded in lots. NIFTY lot size = 50 units

So your real max profit:

₹40 × 50 = ₹2,000

### Maximum Loss

= Spread width – premium

= 100 – 40 = ₹60 per unit

Actual max loss with lot size:

₹60 × 50 = ₹3,000

## When Does Loss Begin?

- Loss starts if NIFTY moves outside the breakevens (22,240 or 21,760)
- Maximum loss only happens if NIFTY crosses beyond the long strikes (22,300 or 21,700)

This distinction helps traders understand the difference between starting to lose vs. hitting max loss.

## Iron Condor Profit and Loss

The Iron Condor has a limited profit and a limited loss, which makes it a popular non-directional options strategy among Indian index traders. Since the structure uses both short and long options, traders know their risk and reward in advance.

- **Maximum Profit:** The maximum profit in an Iron Condor is the net premium received when setting up the trade. This happens when the index stays between the two short strikes until expiry. In this situation, all four options expire worthless, and you keep the entire premium.
- In the Indian market, traders often aim for this during weeks when NIFTY or BANK NIFTY is expected to remain range-bound. Time decay works strongly in their favour, especially during weekly expiries.
- **Maximum Loss:** The maximum loss occurs when the underlying index moves sharply in one direction of the strategy. Even if the market makes a big move, the loss is capped because the long options act as insurance. The loss is calculated as the difference between the spread width and the net premium received. This defined-risk structure is why margin requirements are much lower compared to naked option selling in India.

## Breakeven Points

There are two breakeven levels, one on the upper side and one on the lower side. These breakeven prices are calculated by adjusting the short strikes with the net credit received. If the market closes near either breakeven point, the strategy may result in a small loss or a minimal profit.

## What Determines Profitability?

The Iron Condor works best when the market remains stable, volatility drops, and time decay accelerates. Profit declines as volatility increases or the market starts trending in one direction. However, as long as the price stays inside the sold strike range, the strategy remains profitable.

## Conclusion

The Iron Condor strategy is one of the most practical, beginner-friendly, probability-based strategies for options traders. With defined risk, limited margin, and high probability of success, it remains a top choice for NIFTY and BANK NIFTY traders who prefer neutral setups.

By understanding lot size, margin, breakevens, and simple adjustments, beginners can use iron condors more confidently and safely.

Important Disclaimer: Options trading involves significant risk and is not suitable for all investors. The strategies discussed in this guide are for educational purposes only and do not constitute investment advice. Before executing any trades, please consult a certified financial advisor and ensure you understand the risks involved, particularly the potential for loss of capital.

## FAQs

### 1\. Is the iron condor a good strategy?

Yes. It is ideal for sideways markets and offers high probability with limited risk, making it beginner-friendly.

### 2\. What is the success rate of an iron condor?

Typically 60%–80%, depending on strike selection, volatility, and risk management. Note: This range is widely cited for structured credit strategies.

### 3\. When should I exit an iron condor?

Common exit rules:

- When 50–70% of the premium is captured
- If price breaks beyond breakevens
- If volatility spikes unexpectedly

### 4\. Is an iron condor better than a credit spread?

An iron condor provides:

- Higher probability
- More premium
- A wider range

But a single credit spread requires less margin and is simpler. Many beginners start with credit spreads and later graduate to iron condors.

### 5\. How much can I lose on an iron condor?

Maximum loss = (Spread width – Premium Received) × Lot Size

Loss is capped and known in advance.

### 6\. What is the difference between an iron fly and an iron condor?

Iron Fly:

Short ATM options → Higher reward, lower probability

Iron Condor:

Short OTM options → Lower reward, higher probability

- [Facebook](https://www.facebook.com/ChoiceHQ/)
- [Facebook](https://api.whatsapp.com/send?text=38)
- [Linkedin](https://www.linkedin.com/company/choice-broking/)
- [Twitter](https://twitter.com/ChoiceHQ_Social)

### Recommended for you

![loading](https://choiceindia.com/blogs/content/images/2023/11/indian-stock-market-prediction-for-next-week.jpg)

#### Indian Stock Market Prediction For Next Week

![loading](https://choiceindia.com/blogs/content/images/2023/02/copper-price-forecast-for-next-week.png)

#### Copper Price Forecast for Next Week

![loading](https://choiceindia.com/blogs/content/images/2026/03/what-is-upper-circuit-and-lower-circuit.jpg)

#### What Are The Upper Circuit and Lower Circuit?

![loading](https://choiceindia.com/blogs/content/images/2026/03/what-is-difference-between-small-cap-mid-cap-and-large-cap.jpg)

#### What Is The Difference Between Large Cap, Mid Cap, and Small Cap?

;

reCAPTCHA