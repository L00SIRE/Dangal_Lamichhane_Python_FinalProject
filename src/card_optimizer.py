from typing import Dict, List, Optional, Tuple

import pandas as pd


class CreditCard:
    """
    Represents a credit card with specific reward rates.
    Demonstrates OOP encapsulation and class design.
    """
    
    def __init__(
        self, 
        name: str, 
        base_rate: float, 
        category_rates: Dict[str, float],
        annual_fee: float = 0.0
    ) -> None:
        """
        Initialize a credit card.
        
        Args:
            name: Name of the credit card
            base_rate: Base cashback rate (as decimal, e.g., 0.01 for 1%)
            category_rates: Dictionary mapping category names to bonus rates
            annual_fee: Annual fee in dollars
        """
        self._name: str = name
        self._base_rate: float = base_rate
        self._category_rates: Dict[str, float] = category_rates
        self._annual_fee: float = annual_fee
    
    def get_name(self) -> str:
        """Get the card name."""
        return self._name
    
    def calculate_rewards(
        self, 
        amount: float, 
        category: Optional[str] = None
    ) -> float:
        """
        Calculate rewards for a transaction.
        
        Args:
            amount: Transaction amount
            category: Transaction category (optional)
            
        Returns:
            Reward amount in dollars
        """
        if category and category.lower() in self._category_rates:
            rate = self._base_rate + self._category_rates[category.lower()]
        else:
            rate = self._base_rate
        
        return amount * rate
    
    def get_annual_fee(self) -> float:
        """Get the annual fee."""
        return self._annual_fee


class CardOptimizer:
    """
    Optimizes credit card selection based on spending patterns.
    Simulates different credit card types and calculates optimal rewards.
    """
    
    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the optimizer with transaction data.
        
        Args:
            dataframe: DataFrame containing transaction data
        """
        self._df: pd.DataFrame = dataframe.copy()
        self._cards: List[CreditCard] = self._create_default_cards()
    
    def _create_default_cards(self) -> List[CreditCard]:
        """
        Create default credit card options.
        Includes Travel cards and Standard cards as mentioned in abstract.
        
        Returns:
            List of CreditCard objects
        """
        cards: List[CreditCard] = []
        
        # Travel Card - Higher rewards for travel categories
        travel_card = CreditCard(
            name="Travel Rewards Card",
            base_rate=0.01,  # 1% base
            category_rates={
                "travel": 0.04,  # 5% total on travel
                "airline": 0.04,
                "hotel": 0.03,   # 4% total on hotels
                "restaurant": 0.02,  # 3% total on restaurants
            },
            annual_fee=95.0
        )
        cards.append(travel_card)
        
        # Standard Cashback Card - Flat rate
        standard_card = CreditCard(
            name="Standard Cashback Card",
            base_rate=0.02,  # 2% flat rate
            category_rates={},
            annual_fee=0.0
        )
        cards.append(standard_card)
        
        # Category Card - Bonus on specific categories
        category_card = CreditCard(
            name="Category Bonus Card",
            base_rate=0.01,  # 1% base
            category_rates={
                "grocery": 0.05,  # 6% total on groceries
                "gas": 0.02,       # 3% total on gas
                "dining": 0.02,     # 3% total on dining
            },
            annual_fee=0.0
        )
        cards.append(category_card)
        
        return cards
    
    def add_card(self, card: CreditCard) -> None:
        """
        Add a custom credit card to compare.
        
        Args:
            card: CreditCard object to add
        """
        self._cards.append(card)
    
    def calculate_total_rewards(
        self, 
        card: CreditCard,
        amount_column: str,
        category_column: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total rewards for a card based on spending patterns.
        
        Args:
            card: CreditCard object
            amount_column: Name of the amount column
            category_column: Name of the category column (optional)
            
        Returns:
            Tuple of (total_rewards, category_breakdown)
        """
        if amount_column not in self._df.columns:
            return (0.0, {})
        
        total_rewards = 0.0
        category_breakdown: Dict[str, float] = {}
        
        for _, row in self._df.iterrows():
            amount = float(row[amount_column])
            if pd.isna(amount) or amount <= 0:
                continue
            
            category = None
            if category_column and category_column in self._df.columns:
                category = str(row[category_column]).lower() if pd.notna(row[category_column]) else None
            
            reward = card.calculate_rewards(amount, category)
            total_rewards += reward
            
            # Track by category
            cat_key = category if category else "other"
            if cat_key not in category_breakdown:
                category_breakdown[cat_key] = 0.0
            category_breakdown[cat_key] += reward
        
        # Subtract annual fee
        net_rewards = total_rewards - card.get_annual_fee()
        
        return (net_rewards, category_breakdown)
    
    def find_optimal_card(
        self,
        amount_column: str,
        category_column: Optional[str] = None
    ) -> Tuple[CreditCard, float, Dict[str, float]]:
        """
        Find the credit card that provides the most rewards.
        
        Args:
            amount_column: Name of the amount column
            category_column: Name of the category column (optional)
            
        Returns:
            Tuple of (best_card, rewards_amount, category_breakdown)
        """
        best_card: Optional[CreditCard] = None
        best_rewards = float('-inf')
        best_breakdown: Dict[str, float] = {}
        
        for card in self._cards:
            rewards, breakdown = self.calculate_total_rewards(
                card, amount_column, category_column
            )
            if rewards > best_rewards:
                best_rewards = rewards
                best_card = card
                best_breakdown = breakdown
        
        if best_card is None:
            raise ValueError("No cards available for comparison")
        
        return (best_card, best_rewards, best_breakdown)
    
    def compare_all_cards(
        self,
        amount_column: str,
        category_column: Optional[str] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Compare all cards and return results.
        
        Args:
            amount_column: Name of the amount column
            category_column: Name of the category column (optional)
            
        Returns:
            List of tuples: (card_name, gross_rewards, net_rewards)
        """
        results: List[Tuple[str, float, float]] = []
        
        for card in self._cards:
            rewards, _ = self.calculate_total_rewards(
                card, amount_column, category_column
            )
            gross_rewards = rewards + card.get_annual_fee()
            net_rewards = rewards
            results.append((card.get_name(), gross_rewards, net_rewards))
        
        # Sort by net rewards descending
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def get_spending_by_category(
        self,
        amount_column: str,
        category_column: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate total spending by category.
        
        Args:
            amount_column: Name of the amount column
            category_column: Name of the category column (optional)
            
        Returns:
            Dictionary mapping categories to total spending
        """
        if amount_column not in self._df.columns:
            return {}
        
        spending: Dict[str, float] = {}
        
        for _, row in self._df.iterrows():
            amount = float(row[amount_column])
            if pd.isna(amount) or amount <= 0:
                continue
            
            if category_column and category_column in self._df.columns:
                category = str(row[category_column]).lower() if pd.notna(row[category_column]) else "other"
            else:
                category = "all"
            
            if category not in spending:
                spending[category] = 0.0
            spending[category] += amount
        
        return spending
