import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Interaction patterns with realistic ratios
const INTERACTION_PATTERNS = [
  { name: 'Light Browsing', views: 15, carts: 2, purchases: 0 },
  { name: 'Window Shopping', views: 25, carts: 5, purchases: 1 },
  { name: 'Serious Shopper', views: 35, carts: 8, purchases: 3 },
  { name: 'Power User', views: 50, carts: 12, purchases: 5 },
  { name: 'Frequent Buyer', views: 40, carts: 15, purchases: 8 }
];

function App() {
  const [userProfile, setUserProfile] = useState({
    age: 30,
    gender: 'male',
    income: 50000,
    interaction_history: []
  });
  
  const [recommendationType, setRecommendationType] = useState('hybrid');
  const [numRecommendations, setNumRecommendations] = useState(100);
  const [collaborativeWeight, setCollaborativeWeight] = useState(0.7);
  
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Pagination for recommendations
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(20); // Show 20 recommendations per page
  
  const [sampleItems, setSampleItems] = useState([]);
  const [interactions, setInteractions] = useState([]);
  
  const [expandedInteraction, setExpandedInteraction] = useState(null);
  const [selectedPattern, setSelectedPattern] = useState(null);
  
  // Real user data states
  const [realUsers, setRealUsers] = useState([]);
  const [selectedRealUser, setSelectedRealUser] = useState(null);
  const [datasetSummary, setDatasetSummary] = useState(null);
  const [useRealUsers, setUseRealUsers] = useState(true);
  
  // Expanded interaction states
  const [showUserInteractions, setShowUserInteractions] = useState(false);
  const [userInteractionDetails, setUserInteractionDetails] = useState(null);
  const [loadingInteractions, setLoadingInteractions] = useState(false);

  // Category selection states
  const [selectedCategory, setSelectedCategory] = useState('');
  const [availableCategories, setAvailableCategories] = useState([]);

  // Performance monitoring states
  const [requestHistory, setRequestHistory] = useState([]);
  const [showPerformanceDetails, setShowPerformanceDetails] = useState(false);

  // Random behavioral pattern states
  const [randomBehavioralPatterns, setRandomBehavioralPatterns] = useState([]);
  const [selectedBehavioralPattern, setSelectedBehavioralPattern] = useState(null);

  // Similar items states
  const [similarItems, setSimilarItems] = useState([]);
  const [selectedItemForSimilarity, setSelectedItemForSimilarity] = useState(null);
  const [showSimilarItems, setShowSimilarItems] = useState(false);
  const [loadingSimilarItems, setLoadingSimilarItems] = useState(false);

  // Load sample items and real users on component mount
  useEffect(() => {
    fetchSampleItems();
    fetchRealUsers();
    fetchDatasetSummary();
    fetchRandomBehavioralPatterns();
  }, []);

  // Extract categories when user interactions change
  useEffect(() => {
    extractCategoriesFromInteractions();
  }, [selectedRealUser, userInteractionDetails, interactions, selectedBehavioralPattern]);

  // Reset category selection when user changes
  useEffect(() => {
    setSelectedCategory('');
    setSelectedBehavioralPattern(null);
  }, [selectedRealUser, useRealUsers]);

  const fetchSampleItems = async () => {
    try {
      // Fetch more items to increase chance of finding behavioral pattern items
      const response = await axios.get(`${API_BASE_URL}/items?limit=200`);
      const items = response.data.items || [];
      setSampleItems(items);
      console.log('Sample items loaded:', items.length, 'items');
      console.log('Sample of loaded items:', items.slice(0, 3));
      
      // If user has already selected a pattern but interactions weren't generated due to missing items,
      // generate them now
      if (selectedPattern && interactions.length === 0 && items.length > 0) {
        console.log('Generating previously selected pattern interactions now that items are loaded');
        generateRealisticInteractions(selectedPattern);
      }
    } catch (error) {
      console.error('Error fetching sample items:', error);
    }
  };


  const fetchRealUsers = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/real-users?count=100&min_interactions=5`);
      setRealUsers(response.data.users || []);
      if (response.data.users && response.data.users.length > 0) {
        // Auto-select the first (most active) user
        handleRealUserSelect(response.data.users[0]);
      }
    } catch (error) {
      console.error('Error fetching real users:', error);
      setError('Could not load real users. Using synthetic data mode.');
      setUseRealUsers(false);
    }
  };

  const fetchDatasetSummary = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/dataset-summary`);
      setDatasetSummary(response.data);
    } catch (error) {
      console.error('Error fetching dataset summary:', error);
    }
  };

  const fetchRandomBehavioralPatterns = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/behavioral-patterns?count=100&min_interactions=3`);
      // Extract enriched behavioral patterns with item details
      const patterns = response.data.patterns.map(pattern => ({
        id: pattern.user_id,
        user_id: pattern.user_id,
        summary: pattern.summary,
        pattern: pattern.interaction_pattern,
        stats: pattern.interaction_stats,
        interactionHistory: pattern.enriched_interactions.map(item => item.product_id), // Keep backward compatibility
        enrichedInteractions: pattern.enriched_interactions // New enriched data
      }));
      setRandomBehavioralPatterns(patterns);
    } catch (error) {
      console.error('Error fetching enriched behavioral patterns:', error);
    }
  };

  const fetchSimilarItems = async (itemId, numRecommendations = 10) => {
    try {
      console.log('Fetching similar items for item', itemId);
      setLoadingSimilarItems(true);
      
      const response = await axios.post(`${API_BASE_URL}/item-similarity`, {
        item_id: itemId,
        num_recommendations: numRecommendations
      });
      
      console.log('Received', response.data.length, 'similar items');
      setSimilarItems(response.data);
      return response.data;
    } catch (error) {
      console.error('Error fetching similar items:', error);
      setSimilarItems([]);
      return [];
    } finally {
      setLoadingSimilarItems(false);
    }
  };

  const handleItemClick = async (itemId, itemInfo) => {
    console.log('Item clicked:', itemId, itemInfo);
    
    // Set the selected item info
    setSelectedItemForSimilarity({
      id: itemId,
      info: itemInfo
    });
    
    // Show the modal first
    setShowSimilarItems(true);
    
    // Fetch similar items
    await fetchSimilarItems(itemId);
  };

  const closeSimilarItems = () => {
    setShowSimilarItems(false);
    setSimilarItems([]);
    setSelectedItemForSimilarity(null);
  };

  // Extract hierarchical categories from user interactions
  const extractCategoriesFromInteractions = () => {
    let categories = new Set();
    
    // From real user interactions
    if (selectedRealUser && userInteractionDetails && userInteractionDetails.timeline) {
      userInteractionDetails.timeline.forEach(interaction => {
        if (interaction.category_code) {
          const categoryParts = interaction.category_code.split('.');
          // Add hierarchical categories: electronics, electronics.smartphone, electronics.smartphone.samsung
          for (let i = 1; i <= categoryParts.length; i++) {
            categories.add(categoryParts.slice(0, i).join('.'));
          }
        }
      });
    }
    
    // From behavioral patterns (use enriched interactions)
    if (selectedBehavioralPattern && selectedBehavioralPattern.enrichedInteractions) {
      selectedBehavioralPattern.enrichedInteractions.forEach(item => {
        if (item && item.category_code && item.category_code !== 'Unknown') {
          const categoryParts = item.category_code.split('.');
          for (let i = 1; i <= categoryParts.length; i++) {
            categories.add(categoryParts.slice(0, i).join('.'));
          }
        }
      });
    }
    
    // From synthetic interactions
    interactions.forEach(interaction => {
      if (interaction.category) {
        const categoryParts = interaction.category.split('.');
        for (let i = 1; i <= categoryParts.length; i++) {
          categories.add(categoryParts.slice(0, i).join('.'));
        }
      }
    });
    
    // Convert to sorted array
    const sortedCategories = Array.from(categories).sort();
    setAvailableCategories(sortedCategories);
  };

  // Record performance metrics for requests
  const recordRequestMetrics = (requestType, startTime, endTime, requestDetails, responseDetails) => {
    const duration = endTime - startTime;
    const newRecord = {
      id: Date.now(),
      timestamp: new Date(startTime),
      requestType,
      duration,
      requestDetails: {
        userAge: requestDetails.user_profile?.age,
        userGender: requestDetails.user_profile?.gender,
        numRecommendations: requestDetails.num_recommendations,
        recommendationType: requestDetails.recommendation_type,
        selectedCategory: requestDetails.selected_category || 'None',
        interactionHistorySize: requestDetails.user_profile?.interaction_history?.length || 0
      },
      responseDetails: {
        recommendationCount: responseDetails.recommendations?.length || 0,
        totalCount: responseDetails.total_count || 0
      }
    };
    
    setRequestHistory(prev => [newRecord, ...prev.slice(0, 19)]); // Keep last 20 records
  };

  // Apply selected behavioral pattern to custom user
  const applyBehavioralPattern = (pattern) => {
    setSelectedBehavioralPattern(pattern);
    
    // Update user profile with the behavioral pattern's interaction history
    setUserProfile(prev => ({
      ...prev,
      interaction_history: pattern.interactionHistory.slice(0, 50) // Limit to 50 items for performance
    }));
    
    // Clear synthetic interactions since we're using real behavioral data
    setInteractions([]);
    setSelectedPattern(null);
  };

  // Clear behavioral pattern and reset to empty interactions
  const clearBehavioralPattern = () => {
    setSelectedBehavioralPattern(null);
    setUserProfile(prev => ({
      ...prev,
      interaction_history: []
    }));
  };

  const handleProfileChange = (field, value) => {
    setUserProfile(prev => ({
      ...prev,
      [field]: field === 'age' || field === 'income' ? parseInt(value) || 0 : value
    }));
  };

  const handleRealUserSelect = (user) => {
    setSelectedRealUser(user);
    setUserProfile({
      age: user.age,
      gender: user.gender,
      income: user.income,
      interaction_history: user.interaction_history.slice(0, 50) // Limit to 50 items
    });
    // Clear any synthetic interactions and expanded states
    setInteractions([]);
    setSelectedPattern(null);
    setShowUserInteractions(false);
    setUserInteractionDetails(null);
  };

  const fetchUserInteractionDetails = async (userId) => {
    setLoadingInteractions(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/real-users/${userId}`);
      setUserInteractionDetails(response.data);
    } catch (error) {
      console.error('Error fetching user interaction details:', error);
      setError('Could not load user interaction details');
    } finally {
      setLoadingInteractions(false);
    }
  };

  const toggleUserInteractions = async () => {
    if (!showUserInteractions && selectedRealUser && !userInteractionDetails) {
      await fetchUserInteractionDetails(selectedRealUser.user_id);
    }
    setShowUserInteractions(!showUserInteractions);
  };

  const generateTimestamp = (baseTime, offsetHours) => {
    const timestamp = new Date(baseTime.getTime() - (offsetHours * 60 * 60 * 1000));
    return timestamp.toISOString().replace('T', ' ').slice(0, 19);
  };

  const generateRealisticInteractions = (pattern) => {
    if (sampleItems.length === 0) {
      console.log('Sample items not loaded yet, retrying in 500ms...', {
        patternName: pattern.name,
        sampleItemsLength: sampleItems.length
      });
      // Retry after a short delay to allow sampleItems to load
      setTimeout(() => {
        if (sampleItems.length > 0) {
          console.log('Sample items now available, generating interactions...', sampleItems.length);
          generateRealisticInteractions(pattern);
        } else {
          console.log('Sample items still not available after retry');
        }
      }, 500);
      return;
    }

    console.log('Generating realistic interactions with', sampleItems.length, 'sample items for pattern:', pattern.name);
    const newInteractions = [];
    const usedItems = new Set();
    const baseTime = new Date();
    let timeOffset = pattern.views + pattern.carts + pattern.purchases;

    // Generate views (most common)
    for (let i = 0; i < pattern.views; i++) {
      const item = sampleItems[Math.floor(Math.random() * sampleItems.length)];
      newInteractions.push({
        id: Date.now() + Math.random(),
        type: 'view',
        item_id: item.product_id,
        brand: item.brand,
        category: item.category_code,
        price: item.price,
        timestamp: generateTimestamp(baseTime, timeOffset--),
        session_id: `session_${Math.floor(Math.random() * 1000)}`
      });
    }

    // Generate cart additions (medium frequency, can repeat items)
    for (let i = 0; i < pattern.carts; i++) {
      // 70% chance to use an item that was viewed, 30% new item
      let item;
      const viewedItems = newInteractions.filter(int => int.type === 'view');
      if (viewedItems.length > 0 && Math.random() < 0.7) {
        const viewedItem = viewedItems[Math.floor(Math.random() * viewedItems.length)];
        item = sampleItems.find(si => si.product_id === viewedItem.item_id);
      } else {
        item = sampleItems[Math.floor(Math.random() * sampleItems.length)];
      }

      newInteractions.push({
        id: Date.now() + Math.random(),
        type: 'cart',
        item_id: item.product_id,
        brand: item.brand,
        category: item.category_code,
        price: item.price,
        timestamp: generateTimestamp(baseTime, timeOffset--),
        session_id: `session_${Math.floor(Math.random() * 1000)}`,
        quantity: Math.floor(Math.random() * 3) + 1
      });
    }

    // Generate purchases (least common, usually from cart)
    for (let i = 0; i < pattern.purchases; i++) {
      // 80% chance to use an item from cart, 20% direct purchase
      let item;
      const cartItems = newInteractions.filter(int => int.type === 'cart');
      if (cartItems.length > 0 && Math.random() < 0.8) {
        const cartItem = cartItems[Math.floor(Math.random() * cartItems.length)];
        item = sampleItems.find(si => si.product_id === cartItem.item_id);
      } else {
        item = sampleItems[Math.floor(Math.random() * sampleItems.length)];
      }

      const quantity = Math.floor(Math.random() * 2) + 1;
      newInteractions.push({
        id: Date.now() + Math.random(),
        type: 'purchase',
        item_id: item.product_id,
        brand: item.brand,
        category: item.category_code,
        price: item.price,
        timestamp: generateTimestamp(baseTime, timeOffset--),
        session_id: `session_${Math.floor(Math.random() * 1000)}`,
        quantity: quantity,
        total_amount: (item.price * quantity).toFixed(2)
      });
      
      usedItems.add(item.product_id);
    }

    // Sort by timestamp (most recent first)
    newInteractions.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    setInteractions(newInteractions);
    
    // Update user profile with unique item IDs for recommendations
    const uniqueItemIds = [...usedItems];
    setUserProfile(prev => ({
      ...prev,
      interaction_history: uniqueItemIds
    }));
  };

  const handlePatternSelect = (pattern) => {
    setSelectedPattern(pattern);
    generateRealisticInteractions(pattern);
  };

  const toggleInteractionExpand = (interactionId) => {
    setExpandedInteraction(
      expandedInteraction === interactionId ? null : interactionId
    );
  };

  const clearInteractions = () => {
    setInteractions([]);
    setSelectedPattern(null);
    setUserProfile(prev => ({
      ...prev,
      interaction_history: []
    }));
  };


  const getInteractionCounts = () => {
    const counts = { views: 0, carts: 0, purchases: 0 };
    interactions.forEach(interaction => {
      counts[interaction.type + 's'] = (counts[interaction.type + 's'] || 0) + 1;
    });
    return counts;
  };

  const counts = getInteractionCounts();

  // Calculate category percentages from user interactions
  const getCategoryPercentages = () => {
    console.log('getCategoryPercentages called:', {
      useRealUsers,
      selectedBehavioralPattern: !!selectedBehavioralPattern,
      interactionsLength: interactions.length,
      sampleItemsLength: sampleItems.length
    });

    // For custom users (useRealUsers = false), prioritize custom interactions
    if (!useRealUsers) {
      if (selectedBehavioralPattern) {
        console.log('Processing behavioral pattern categories...');
        
        // Use enriched interactions directly (no need for caching or additional API calls)
        if (selectedBehavioralPattern.enrichedInteractions && selectedBehavioralPattern.enrichedInteractions.length > 0) {
          console.log('Using', selectedBehavioralPattern.enrichedInteractions.length, 'enriched behavioral pattern items');
          
          const categoryCounts = {};
          let totalInteractions = 0;

          selectedBehavioralPattern.enrichedInteractions.forEach(item => {
            if (item && item.category_code && item.category_code !== 'Unknown') {
              const category = item.category_code;
              categoryCounts[category] = (categoryCounts[category] || 0) + 1;
              totalInteractions++;
            }
          });

          console.log('Enriched behavioral pattern results:', { categoryCounts, totalInteractions });

          if (totalInteractions > 0) {
            const categoryPercentages = {};
            Object.keys(categoryCounts).forEach(category => {
              categoryPercentages[category] = ((categoryCounts[category] / totalInteractions) * 100).toFixed(1);
            });
            console.log('Returning enriched behavioral pattern percentages:', categoryPercentages);
            return categoryPercentages;
          }
        } else {
          console.log('No enriched interactions found for behavioral pattern');
        }
      } else if (interactions.length > 0) {
        console.log('Processing synthetic interactions...', interactions);
        // Synthetic interactions categories
        const categoryCounts = {};
        let totalInteractions = 0;

        interactions.forEach(interaction => {
          console.log('Processing interaction:', interaction);
          if (interaction.category && interaction.category !== 'Unknown') {
            const category = interaction.category;
            categoryCounts[category] = (categoryCounts[category] || 0) + 1;
            totalInteractions++;
          }
        });

        console.log('Synthetic interaction results:', { categoryCounts, totalInteractions });

        if (totalInteractions > 0) {
          const categoryPercentages = {};
          Object.keys(categoryCounts).forEach(category => {
            categoryPercentages[category] = ((categoryCounts[category] / totalInteractions) * 100).toFixed(1);
          });
          console.log('Returning synthetic percentages:', categoryPercentages);
          return categoryPercentages;
        }
      }
    }
    
    // For real users (useRealUsers = true), use real user data
    if (useRealUsers && selectedRealUser && userInteractionDetails) {
      // Real user categories from timeline
      const categoryCounts = {};
      let totalInteractions = 0;

      userInteractionDetails.timeline?.forEach(interaction => {
        const category = interaction.category_code || 'Unknown';
        categoryCounts[category] = (categoryCounts[category] || 0) + 1;
        totalInteractions++;
      });

      const categoryPercentages = {};
      Object.keys(categoryCounts).forEach(category => {
        categoryPercentages[category] = ((categoryCounts[category] / totalInteractions) * 100).toFixed(1);
      });

      return categoryPercentages;
    }
    
    console.log('Returning empty object from getCategoryPercentages');
    return {};
  };

  // Calculate recommendation category percentages
  const getRecommendationCategoryPercentages = () => {
    if (!recommendations || recommendations.length === 0) return {};
    
    const recCategoryCounts = {};
    
    recommendations.forEach(rec => {
      const category = rec.item_info?.category_code || 'Unknown';
      recCategoryCounts[category] = (recCategoryCounts[category] || 0) + 1;
    });

    const recCategoryPercentages = {};
    Object.keys(recCategoryCounts).forEach(category => {
      recCategoryPercentages[category] = ((recCategoryCounts[category] / recommendations.length) * 100).toFixed(1);
    });

    return recCategoryPercentages;
  };

  // Use useMemo to ensure React properly tracks dependencies and re-calculates
  const categoryPercentages = React.useMemo(() => {
    console.log('Calculating categoryPercentages with dependencies:', {
      useRealUsers,
      selectedBehavioralPattern: !!selectedBehavioralPattern,
      interactionsLength: interactions.length,
      sampleItemsLength: sampleItems.length,
      hasEnrichedInteractions: !!(selectedBehavioralPattern?.enrichedInteractions?.length)
    });
    return getCategoryPercentages();
  }, [useRealUsers, selectedBehavioralPattern, interactions, sampleItems, selectedRealUser, userInteractionDetails]);
  
  const recommendationCategoryPercentages = React.useMemo(() => {
    console.log('Calculating recommendationCategoryPercentages with', recommendations.length, 'recommendations');
    return getRecommendationCategoryPercentages();
  }, [recommendations]);

  // Pagination logic
  const totalPages = Math.ceil(recommendations.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentRecommendations = recommendations.slice(startIndex, endIndex);

  const goToPage = (page) => {
    setCurrentPage(page);
    // Scroll to recommendations section
    document.querySelector('.recommendations')?.scrollIntoView({ behavior: 'smooth' });
  };

  // Reset pagination when new recommendations are generated
  const getRecommendations = async () => {
    setLoading(true);
    setError(null);
    setCurrentPage(1); // Reset to first page
    
    const startTime = performance.now();
    
    try {
      const requestData = {
        user_profile: userProfile,
        num_recommendations: numRecommendations,
        recommendation_type: recommendationType,
        collaborative_weight: collaborativeWeight,
        selected_category: selectedCategory || null
      };
      
      const response = await axios.post(`${API_BASE_URL}/recommendations`, requestData);
      const endTime = performance.now();
      
      // Record performance metrics
      recordRequestMetrics('recommendations', startTime, endTime, requestData, response.data);
      
      setRecommendations(response.data.recommendations);
    } catch (error) {
      const endTime = performance.now();
      console.error('Error fetching recommendations:', error);
      setError(error.response?.data?.detail || 'Failed to fetch recommendations');
      
      // Record failed request metrics
      recordRequestMetrics('recommendations', startTime, endTime, {
        user_profile: userProfile,
        num_recommendations: numRecommendations,
        recommendation_type: recommendationType,
        selected_category: selectedCategory || null
      }, { error: true });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>Two-Tower Recommendation System Demo</h1>
          <p>Select from {realUsers.length} real users or configure custom demographics to get personalized recommendations</p>
          
          {datasetSummary && (
            <div className="dataset-info">
              📊 Dataset: {datasetSummary.total_users?.toLocaleString()} users, {datasetSummary.total_interactions?.toLocaleString()} interactions | 
              👥 Demographics: Avg age {datasetSummary.demographics?.avg_age}, avg income ${datasetSummary.demographics?.avg_income?.toLocaleString()}
            </div>
          )}
        </header>

        {/* Real User Selector */}
        {useRealUsers && realUsers.length > 0 && (
          <div className="real-user-selector">
            <h2>Real User Selection</h2>
            <div className="user-selector-controls">
              <label htmlFor="realUserSelect">Choose from {realUsers.length} real users:</label>
              <select 
                id="realUserSelect"
                value={selectedRealUser?.user_id || ''}
                onChange={(e) => {
                  const userId = parseInt(e.target.value);
                  const user = realUsers.find(u => u.user_id === userId);
                  if (user) handleRealUserSelect(user);
                }}
              >
                <option value="">Select a real user...</option>
                {realUsers.map((user, index) => (
                  <option key={user.user_id} value={user.user_id}>
                    #{index + 1}: {user.summary} - {user.interaction_pattern}
                  </option>
                ))}
              </select>
              
              <button 
                onClick={() => setUseRealUsers(false)}
                className="btn btn-secondary"
                style={{marginLeft: '10px'}}
              >
                Use Custom User Instead
              </button>
            </div>

            {selectedRealUser && (
              <div className="selected-real-user">
                <h3>Selected User: {selectedRealUser.user_id}</h3>
                <div className="real-user-stats">
                  <div className="user-stat">
                    <span className="stat-label">Demographics:</span>
                    <span className="stat-value">{selectedRealUser.age}yr {selectedRealUser.gender}, ${selectedRealUser.income.toLocaleString()}</span>
                  </div>
                  <div className="user-stat">
                    <span className="stat-label">Behavior Pattern:</span>
                    <span className="stat-value">{selectedRealUser.interaction_pattern}</span>
                  </div>
                  <div className="user-stat">
                    <span className="stat-label">Interactions:</span>
                    <span className="stat-value">
                      {selectedRealUser.interaction_stats.total_interactions} total 
                      ({selectedRealUser.interaction_stats.views} views, {selectedRealUser.interaction_stats.cart_adds} carts, {selectedRealUser.interaction_stats.purchases} purchases)
                    </span>
                  </div>
                  <div className="user-stat">
                    <span className="stat-label">History:</span>
                    <span className="stat-value">{selectedRealUser.interaction_stats.unique_items} unique items</span>
                  </div>
                </div>
                
                <button 
                  onClick={toggleUserInteractions}
                  className="btn btn-info expand-interactions-btn"
                  disabled={loadingInteractions}
                >
                  {loadingInteractions ? 'Loading...' : showUserInteractions ? 'Hide Interaction Timeline' : 'Show All Interactions Timeline'}
                </button>

                {showUserInteractions && userInteractionDetails && (
                  <div className="user-interactions-timeline">
                    <h4>Complete Interaction Timeline</h4>
                    <div className="timeline-stats">
                      <span><strong>Total Events:</strong> {userInteractionDetails.total_interactions}</span>
                      <span><strong>Pattern:</strong> {userInteractionDetails.interaction_pattern}</span>
                      <span><strong>Breakdown:</strong> {userInteractionDetails.breakdown.views} views, {userInteractionDetails.breakdown.cart_adds} carts, {userInteractionDetails.breakdown.purchases} purchases</span>
                    </div>
                    
                    <div className="interactions-list">
                      <h5>Recent Interactions (Last {userInteractionDetails.timeline?.length || 0} events):</h5>
                      {userInteractionDetails.timeline?.map((interaction, index) => (
                        <div key={index} className="interaction-timeline-item">
                          <div className="interaction-timeline-time">
                            {new Date(interaction.timestamp).toLocaleString()}
                          </div>
                          <div className="interaction-timeline-content">
                            <div className="interaction-main-info">
                              <span className={`interaction-type ${interaction.event_type}`}>
                                {interaction.event_type.toUpperCase()}
                              </span>
                              <span className="interaction-icon">
                                {interaction.event_type === 'purchase' && '💰'}
                                {interaction.event_type === 'cart' && '🛒'}
                                {interaction.event_type === 'view' && '👁️'}
                              </span>
                              <span className="interaction-item-id">
                                Item #{interaction.product_id}
                              </span>
                            </div>
                            <div className="interaction-item-details">
                              <span className="item-brand">
                                <strong>{interaction.brand || 'Unknown Brand'}</strong>
                              </span>
                              <span className="item-category">
                                {interaction.category_code || 'Unknown Category'}
                              </span>
                              <span className="item-price">
                                ${interaction.price ? interaction.price.toFixed(2) : '0.00'}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* User Profile Form */}
        <div className="user-profile-form">
          <h2>User Demographics {useRealUsers && selectedRealUser ? '(From Real User)' : '(Custom)'}</h2>
          
          {!useRealUsers && (
            <button 
              onClick={() => {
                setUseRealUsers(true);
                if (realUsers.length > 0) handleRealUserSelect(realUsers[0]);
              }}
              className="btn btn-secondary"
              style={{marginBottom: '15px'}}
            >
              Switch to Real Users
            </button>
          )}
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="age">Age:</label>
              <input
                type="number"
                id="age"
                value={userProfile.age}
                onChange={(e) => handleProfileChange('age', e.target.value)}
                min="18"
                max="100"
                disabled={useRealUsers && selectedRealUser}
                style={{backgroundColor: useRealUsers && selectedRealUser ? '#f5f5f5' : 'white'}}
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="gender">Gender:</label>
              <select
                id="gender"
                value={userProfile.gender}
                onChange={(e) => handleProfileChange('gender', e.target.value)}
                disabled={useRealUsers && selectedRealUser}
                style={{backgroundColor: useRealUsers && selectedRealUser ? '#f5f5f5' : 'white'}}
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="income">Annual Income ($):</label>
              <input
                type="number"
                id="income"
                value={userProfile.income}
                onChange={(e) => handleProfileChange('income', e.target.value)}
                min="0"
                step="1000"
                disabled={useRealUsers && selectedRealUser}
                style={{backgroundColor: useRealUsers && selectedRealUser ? '#f5f5f5' : 'white'}}
              />
            </div>
          </div>
        </div>

        {/* Random Behavioral Patterns for Custom Users */}
        {!useRealUsers && (
          <div className="random-behavioral-patterns">
            <h2>Real User Behavioral Patterns</h2>
            <p>Apply real user interaction patterns to your custom demographics</p>
            
            <div className="behavioral-pattern-controls">
              <div className="pattern-refresh-section">
                <button 
                  onClick={fetchRandomBehavioralPatterns}
                  className="btn btn-secondary"
                  disabled={!randomBehavioralPatterns.length && false} // Always enabled
                >
                  🔄 Refresh Random Patterns ({randomBehavioralPatterns.length})
                </button>
                
                {selectedBehavioralPattern && (
                  <button 
                    onClick={clearBehavioralPattern}
                    className="btn btn-outline"
                    style={{marginLeft: '10px'}}
                  >
                    Clear Pattern
                  </button>
                )}
              </div>

              {selectedBehavioralPattern && (
                <div className="selected-behavioral-pattern">
                  <h3>Applied Behavioral Pattern</h3>
                  <div className="pattern-info">
                    <div className="pattern-summary">
                      <strong>Pattern:</strong> {selectedBehavioralPattern.pattern}
                    </div>
                    <div className="pattern-stats">
                      <span><strong>Views:</strong> {selectedBehavioralPattern.stats.views}</span>
                      <span><strong>Cart Adds:</strong> {selectedBehavioralPattern.stats.cart_adds}</span>
                      <span><strong>Purchases:</strong> {selectedBehavioralPattern.stats.purchases}</span>
                    </div>
                    <div className="pattern-items">
                      <strong>Interaction History:</strong> {selectedBehavioralPattern.interactionHistory.length} items 
                      (using {Math.min(50, selectedBehavioralPattern.interactionHistory.length)} for recommendations)
                    </div>
                  </div>
                </div>
              )}

              {randomBehavioralPatterns.length > 0 && !selectedBehavioralPattern && (
                <div className="behavioral-pattern-grid">
                  <h3>Available Patterns ({randomBehavioralPatterns.length})</h3>
                  <div className="pattern-grid">
                    {randomBehavioralPatterns.slice(0, 12).map((pattern) => (
                      <div key={pattern.id} className="pattern-card" onClick={() => applyBehavioralPattern(pattern)}>
                        <div className="pattern-card-header">
                          <div className="pattern-type">{pattern.pattern}</div>
                        </div>
                        <div className="pattern-card-stats">
                          <div className="pattern-stat">
                            <span className="stat-number">{pattern.stats.views}</span>
                            <span className="stat-label">Views</span>
                          </div>
                          <div className="pattern-stat">
                            <span className="stat-number">{pattern.stats.cart_adds}</span>
                            <span className="stat-label">Carts</span>
                          </div>
                          <div className="pattern-stat">
                            <span className="stat-number">{pattern.stats.purchases}</span>
                            <span className="stat-label">Purchases</span>
                          </div>
                        </div>
                        <div className="pattern-card-summary">
                          {pattern.stats.total_interactions} total interactions
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {randomBehavioralPatterns.length > 12 && (
                    <div className="pattern-grid-more">
                      <p>Showing 12 of {randomBehavioralPatterns.length} patterns. Click refresh for different patterns.</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Interaction Patterns */}
        <div className="interaction-patterns">
          {useRealUsers && selectedRealUser ? (
            <>
              <h2>Real User Interaction History</h2>
              <p>This user has genuine interaction history from the dataset - no synthetic patterns needed.</p>
              
              <div className="real-interaction-summary">
                <div className="summary-card views">
                  <div className="summary-number">{selectedRealUser.interaction_stats.views}</div>
                  <div className="summary-label">Views</div>
                </div>
                <div className="summary-card carts">
                  <div className="summary-number">{selectedRealUser.interaction_stats.cart_adds}</div>
                  <div className="summary-label">Cart Adds</div>
                </div>
                <div className="summary-card purchases">
                  <div className="summary-number">{selectedRealUser.interaction_stats.purchases}</div>
                  <div className="summary-label">Purchases</div>
                </div>
              </div>

              <div className="real-history-info">
                <p><strong>Pattern:</strong> {selectedRealUser.interaction_pattern}</p>
                <p><strong>Total Interactions:</strong> {selectedRealUser.interaction_stats.total_interactions}</p>
                <p><strong>Unique Items:</strong> {selectedRealUser.interaction_stats.unique_items}</p>
                <p><strong>Items in History:</strong> {userProfile.interaction_history.length} (showing up to 50 most recent)</p>
              </div>

              {/* Category Analysis Columns */}
              {(Object.keys(categoryPercentages).length > 0 || Object.keys(recommendationCategoryPercentages).length > 0) && (
                <div className="category-analysis">
                  <h4>Category Analysis</h4>
                  <div className="category-columns">
                    
                    {/* User's Interacted Categories */}
                    {Object.keys(categoryPercentages).length > 0 && (
                      <div className="category-column">
                        <h5>👁️ User's Category Interests</h5>
                        <div className="category-percentages">
                          {Object.entries(categoryPercentages)
                            .sort((a, b) => parseFloat(b[1]) - parseFloat(a[1]))
                            .slice(0, 5)
                            .map(([category, percentage]) => (
                              <div key={category} className="category-item">
                                <div className="category-bar-container">
                                  <div 
                                    className="category-bar user-category"
                                    style={{ width: `${Math.max(parseFloat(percentage), 5)}%` }}
                                  ></div>
                                </div>
                                <span className="category-label">{category.replace('_', ' ')}</span>
                                <span className="category-percent">{percentage}%</span>
                              </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Recommendation Categories */}
                    {Object.keys(recommendationCategoryPercentages).length > 0 && (
                      <div className="category-column">
                        <h5>🎯 Recommendation Categories</h5>
                        <div className="category-percentages">
                          {Object.entries(recommendationCategoryPercentages)
                            .sort((a, b) => parseFloat(b[1]) - parseFloat(a[1]))
                            .map(([category, percentage]) => {
                              const userPercentage = categoryPercentages[category] || 0;
                              const isMatch = parseFloat(userPercentage) > 0;
                              
                              return (
                                <div key={category} className={`category-item ${isMatch ? 'matched' : 'new'}`}>
                                  <div className="category-bar-container">
                                    <div 
                                      className={`category-bar ${isMatch ? 'rec-category-matched' : 'rec-category-new'}`}
                                      style={{ width: `${Math.max(parseFloat(percentage), 5)}%` }}
                                    ></div>
                                  </div>
                                  <span className="category-label">{category.replace('_', ' ')}</span>
                                  <span className="category-percent">{percentage}%</span>
                                  {isMatch && <span className="match-indicator">✓</span>}
                                </div>
                              );
                            })}
                        </div>
                      </div>
                    )}

                  </div>

                  {/* Category Match Analysis */}
                  {Object.keys(categoryPercentages).length > 0 && Object.keys(recommendationCategoryPercentages).length > 0 && (
                    <div className="category-match-summary">
                      <p>
                        <strong>Category Alignment:</strong> {
                          Object.keys(recommendationCategoryPercentages).filter(cat => 
                            parseFloat(categoryPercentages[cat] || 0) > 0
                          ).length
                        } of {Object.keys(recommendationCategoryPercentages).length} recommended categories match user interests
                        <span className="match-legend">
                          <span className="legend-item"><span className="legend-dot matched"></span> Matches user interest</span>
                          <span className="legend-item"><span className="legend-dot new"></span> New category exploration</span>
                        </span>
                      </p>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <>
              <h2>Custom User Interaction History</h2>
              <p>Using synthetic patterns or behavioral patterns to simulate user behavior</p>
              
              {/* Custom User Interaction Summary - Similar to Real User Summary */}
              {(selectedBehavioralPattern || interactions.length > 0) && (
                <div className="custom-interaction-summary">
                  {selectedBehavioralPattern ? (
                    <>
                      <div className="summary-card views">
                        <div className="summary-number">{selectedBehavioralPattern.stats.views}</div>
                        <div className="summary-label">Views</div>
                      </div>
                      <div className="summary-card carts">
                        <div className="summary-number">{selectedBehavioralPattern.stats.cart_adds}</div>
                        <div className="summary-label">Cart Adds</div>
                      </div>
                      <div className="summary-card purchases">
                        <div className="summary-number">{selectedBehavioralPattern.stats.purchases}</div>
                        <div className="summary-label">Purchases</div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="summary-card views">
                        <div className="summary-number">{counts.views || 0}</div>
                        <div className="summary-label">Views</div>
                      </div>
                      <div className="summary-card carts">
                        <div className="summary-number">{counts.carts || 0}</div>
                        <div className="summary-label">Cart Adds</div>
                      </div>
                      <div className="summary-card purchases">
                        <div className="summary-number">{counts.purchases || 0}</div>
                        <div className="summary-label">Purchases</div>
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* Custom History Info - Similar to Real User Info */}
              {(selectedBehavioralPattern || interactions.length > 0) && (
                <div className="custom-history-info">
                  {selectedBehavioralPattern ? (
                    <>
                      <p><strong>Pattern:</strong> {selectedBehavioralPattern.pattern}</p>
                      <p><strong>Total Interactions:</strong> {selectedBehavioralPattern.stats.total_interactions}</p>
                      <p><strong>Unique Items:</strong> {selectedBehavioralPattern.interactionHistory.length}</p>
                      <p><strong>Items in History:</strong> {userProfile.interaction_history.length} (using up to 50 most recent)</p>
                    </>
                  ) : (
                    <>
                      <p><strong>Pattern:</strong> {selectedPattern?.name || 'Custom Synthetic'}</p>
                      <p><strong>Total Interactions:</strong> {interactions.length}</p>
                      <p><strong>Unique Items:</strong> {new Set(interactions.map(i => i.item_id)).size}</p>
                      <p><strong>Items in History:</strong> {userProfile.interaction_history.length}</p>
                    </>
                  )}
                </div>
              )}
              
              {/* Category Analysis for Custom Users */}
              {(selectedBehavioralPattern || interactions.length > 0 || userProfile.interaction_history.length > 0) && (
                <div 
                  key={`category-analysis-${interactions.length}-${selectedBehavioralPattern?.id || 'none'}-${sampleItems.length}`}
                  className="category-analysis"
                >
                  <h4>Category Analysis</h4>
                  
                  {/* User Interests vs Recommendations Comparison */}
                  <div className="category-comparison">
                    <div className="category-columns">
                      
                      {/* User's Interacted Categories */}
                      <div className="category-column">
                        <h5>👁️ User's Category Interests</h5>
                        <div className="category-percentages">
                          {Object.keys(categoryPercentages).length > 0 ? (
                            Object.entries(categoryPercentages)
                              .sort((a, b) => parseFloat(b[1]) - parseFloat(a[1]))
                              .slice(0, 5)
                              .map(([category, percentage]) => (
                                <div key={category} className="category-item">
                                  <div className="category-bar-container">
                                    <div 
                                      className="category-bar user-category"
                                      style={{ width: `${Math.max(parseFloat(percentage), 5)}%` }}
                                    ></div>
                                  </div>
                                  <span className="category-label">{category.replace(/\./g, ' > ')}</span>
                                  <span className="category-percent">{percentage}%</span>
                                </div>
                            ))
                          ) : (
                            <div className="category-loading">
                              <p>Processing interaction categories...</p>
                              <small>
                                Debug: useRealUsers={String(useRealUsers)}, 
                                behavioral={String(!!selectedBehavioralPattern)}, 
                                interactions={interactions.length}, 
                                sampleItems={sampleItems.length}
                                <br />
                                categoryPercentages keys: {Object.keys(categoryPercentages).join(', ') || 'none'}
                                <br />
                                Last updated: {new Date().toLocaleTimeString()}
                                <br />
                                {selectedBehavioralPattern && sampleItems.length === 0 && "Loading item catalog..."}
                                {selectedBehavioralPattern && sampleItems.length > 0 && "Behavioral pattern loaded, processing categories..."}
                                {interactions.length === 0 && !selectedBehavioralPattern && "No interactions selected yet"}
                                {interactions.length > 0 && !selectedBehavioralPattern && "Analyzing synthetic interactions..."}
                                {interactions.length > 0 && !selectedBehavioralPattern && sampleItems.length === 0 && " (No sample items available)"}
                              </small>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Recommendation Categories */}
                      {Object.keys(recommendationCategoryPercentages).length > 0 && (
                        <div className="category-column">
                          <h5>🎯 Recommendation Categories</h5>
                          <div className="category-percentages">
                            {Object.entries(recommendationCategoryPercentages)
                              .sort((a, b) => parseFloat(b[1]) - parseFloat(a[1]))
                              .map(([category, percentage]) => {
                                const userPercentage = categoryPercentages[category] || 0;
                                const isMatch = parseFloat(userPercentage) > 0;
                                
                                return (
                                  <div key={category} className={`category-item ${isMatch ? 'matched' : 'new'}`}>
                                    <div className="category-bar-container">
                                      <div 
                                        className={`category-bar ${isMatch ? 'rec-category-matched' : 'rec-category-new'}`}
                                        style={{ width: `${Math.max(parseFloat(percentage), 5)}%` }}
                                      ></div>
                                    </div>
                                    <span className="category-label">{category.replace(/\./g, ' > ')}</span>
                                    <span className="category-percent">{percentage}%</span>
                                    {isMatch && <span className="match-indicator">✓</span>}
                                  </div>
                                );
                              })}
                          </div>
                        </div>
                      )}

                    </div>

                    {/* Category Match Analysis */}
                    {Object.keys(categoryPercentages).length > 0 && Object.keys(recommendationCategoryPercentages).length > 0 && (
                      <div className="category-match-summary">
                        <p>
                          <strong>Category Alignment:</strong> {
                            Object.keys(recommendationCategoryPercentages).filter(cat => 
                              parseFloat(categoryPercentages[cat] || 0) > 0
                            ).length
                          } of {Object.keys(recommendationCategoryPercentages).length} recommended categories match user interests
                          <span className="match-legend">
                            <span className="legend-item"><span className="legend-dot matched"></span> Matches user interest</span>
                            <span className="legend-item"><span className="legend-dot new"></span> New category exploration</span>
                          </span>
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Custom User Interaction Timeline - Similar to Real User Timeline */}
              {(selectedBehavioralPattern || interactions.length > 0) && (
                <div className="custom-user-interactions">
                  <h4>Custom User Interaction Timeline</h4>
                  
                  {selectedBehavioralPattern && (
                    <div className="behavioral-pattern-timeline">
                      <div className="timeline-stats">
                        <span><strong>Pattern:</strong> {selectedBehavioralPattern.pattern}</span>
                        <span><strong>Total Interactions:</strong> {selectedBehavioralPattern.stats.total_interactions}</span>
                        <span><strong>Items:</strong> {selectedBehavioralPattern.interactionHistory.length} unique items</span>
                      </div>
                      
                      <div className="interactions-list">
                        <h5>Behavioral Pattern Items:</h5>
                        {(() => {
                          // Use enriched interactions directly from the behavioral pattern
                          if (selectedBehavioralPattern.enrichedInteractions && selectedBehavioralPattern.enrichedInteractions.length > 0) {
                            return selectedBehavioralPattern.enrichedInteractions.slice(0, 20).map((item, index) => (
                              <div key={index} className="interaction-timeline-item">
                                <div className="interaction-timeline-content">
                                  <div className="interaction-main-info">
                                    <span className="interaction-type behavioral">PATTERN</span>
                                    <span className="interaction-icon">🎭</span>
                                    <span className="interaction-item-id">Item #{item.product_id}</span>
                                  </div>
                                  <div className="interaction-item-details">
                                    <span className="item-brand">
                                      <strong>{item.brand || 'Unknown Brand'}</strong>
                                    </span>
                                    <span className="item-category">
                                      {item.category_code || 'Unknown Category'}
                                    </span>
                                    <span className="item-price">
                                      ${item.price ? item.price.toFixed(2) : '0.00'}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ));
                          } else {
                            return (
                              <div className="loading-behavioral-items">
                                <p>No enriched interactions available</p>
                                <small>This behavioral pattern doesn't have enriched item details</small>
                              </div>
                            );
                          }
                        })()}
                      </div>
                    </div>
                  )}

                  {interactions.length > 0 && (
                    <div className="synthetic-interactions-timeline">
                      <div className="timeline-stats">
                        <span><strong>Synthetic Pattern:</strong> {selectedPattern?.name || 'Custom'}</span>
                        <span><strong>Total Events:</strong> {interactions.length}</span>
                        <span><strong>Breakdown:</strong> {counts.views || 0} views, {counts.carts || 0} carts, {counts.purchases || 0} purchases</span>
                      </div>
                      
                      <div className="interactions-list">
                        <h5>Recent Interactions (Last {Math.min(15, interactions.length)} events):</h5>
                        {interactions.slice(0, 15).map((interaction) => (
                          <div key={interaction.id} className="interaction-timeline-item">
                            <div className="interaction-timeline-time">
                              {new Date(interaction.timestamp).toLocaleString()}
                            </div>
                            <div className="interaction-timeline-content">
                              <div className="interaction-main-info">
                                <span className={`interaction-type ${interaction.type}`}>
                                  {interaction.type.toUpperCase()}
                                </span>
                                <span className="interaction-icon">
                                  {interaction.type === 'purchase' && '💰'}
                                  {interaction.type === 'cart' && '🛒'}
                                  {interaction.type === 'view' && '👁️'}
                                </span>
                                <span className="interaction-item-id">
                                  Item #{interaction.item_id}
                                </span>
                              </div>
                              <div className="interaction-item-details">
                                <span className="item-brand">
                                  <strong>{interaction.brand || 'Unknown Brand'}</strong>
                                </span>
                                <span className="item-category">
                                  {interaction.category || 'Unknown Category'}
                                </span>
                                <span className="item-price">
                                  ${interaction.price ? interaction.price.toFixed(2) : '0.00'}
                                </span>
                                {interaction.quantity && (
                                  <span className="item-quantity">Qty: {interaction.quantity}</span>
                                )}
                                {interaction.total_amount && (
                                  <span className="item-total">Total: ${interaction.total_amount}</span>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              <h3>Synthetic Interaction Patterns</h3>
              <p>Generate realistic user behavior patterns with proportional view, cart, and purchase events</p>
              
              <div className="pattern-buttons">
                {INTERACTION_PATTERNS.map((pattern, index) => (
                  <button
                    key={index}
                    className={`pattern-btn ${selectedPattern?.name === pattern.name ? 'active' : ''}`}
                    onClick={() => handlePatternSelect(pattern)}
                  >
                    {pattern.name}
                    <br />
                    <small>{pattern.views}V • {pattern.carts}C • {pattern.purchases}P</small>
                  </button>
                ))}
                <button
                  className="pattern-btn"
                  onClick={clearInteractions}
                  style={{backgroundColor: '#dc3545', color: 'white', borderColor: '#dc3545'}}
                >
                  Clear All
                </button>
              </div>
            </>
          )}

          {interactions.length > 0 && (
            <>
              <div className="pattern-summary">
                <div className="summary-card views">
                  <div className="summary-number">{counts.views || 0}</div>
                  <div className="summary-label">Views</div>
                </div>
                <div className="summary-card carts">
                  <div className="summary-number">{counts.carts || 0}</div>
                  <div className="summary-label">Cart Adds</div>
                </div>
                <div className="summary-card purchases">
                  <div className="summary-number">{counts.purchases || 0}</div>
                  <div className="summary-label">Purchases</div>
                </div>
              </div>

              <div className="interaction-history">
                <h3>Interaction History ({interactions.length} events)</h3>
                {interactions.map((interaction) => (
                  <div key={interaction.id} className="interaction-item">
                    <div className="interaction-main">
                      <span className={`interaction-type ${interaction.type}`}>
                        {interaction.type}
                      </span>
                      <span className="interaction-details">
                        <strong>{interaction.brand}</strong> - <span className="category-tag">{interaction.category}</span> - ${interaction.price}
                        {interaction.quantity && ` (x${interaction.quantity})`}
                        {interaction.total_amount && ` = $${interaction.total_amount}`}
                      </span>
                      <span style={{fontSize: '12px', color: '#888'}}>
                        {new Date(interaction.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <button
                      className="interaction-expand"
                      onClick={() => toggleInteractionExpand(interaction.id)}
                    >
                      {expandedInteraction === interaction.id ? 'Hide' : 'Details'}
                    </button>
                  </div>
                ))}
                
                {expandedInteraction && (
                  <div className="interaction-expanded">
                    {(() => {
                      const expanded = interactions.find(i => i.id === expandedInteraction);
                      return (
                        <div className="interaction-meta">
                          <div className="interaction-meta-item">
                            <span className="interaction-meta-label">Product ID:</span>
                            <span className="interaction-meta-value">{expanded.item_id}</span>
                          </div>
                          <div className="interaction-meta-item">
                            <span className="interaction-meta-label">Brand:</span>
                            <span className="interaction-meta-value">{expanded.brand}</span>
                          </div>
                          <div className="interaction-meta-item">
                            <span className="interaction-meta-label">Category:</span>
                            <span className="interaction-meta-value">{expanded.category}</span>
                          </div>
                          <div className="interaction-meta-item">
                            <span className="interaction-meta-label">Price:</span>
                            <span className="interaction-meta-value">${expanded.price}</span>
                          </div>
                          <div className="interaction-meta-item">
                            <span className="interaction-meta-label">Timestamp:</span>
                            <span className="interaction-meta-value">{expanded.timestamp}</span>
                          </div>
                          <div className="interaction-meta-item">
                            <span className="interaction-meta-label">Session:</span>
                            <span className="interaction-meta-value">{expanded.session_id}</span>
                          </div>
                          {expanded.quantity && (
                            <div className="interaction-meta-item">
                              <span className="interaction-meta-label">Quantity:</span>
                              <span className="interaction-meta-value">{expanded.quantity}</span>
                            </div>
                          )}
                          {expanded.total_amount && (
                            <div className="interaction-meta-item">
                              <span className="interaction-meta-label">Total Amount:</span>
                              <span className="interaction-meta-value">${expanded.total_amount}</span>
                            </div>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>

            </>
          )}
        </div>

        {/* Category Selection */}
        {availableCategories.length > 0 && (
          <div className="category-selection">
            <h2>Category Filter</h2>
            <div className="category-controls">
              <div className="form-group">
                <label htmlFor="categorySelect">Select Category (optional):</label>
                <select
                  id="categorySelect"
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="category-dropdown"
                >
                  <option value="">All Categories</option>
                  {availableCategories.map((category) => (
                    <option key={category} value={category}>
                      {category.replace(/\./g, ' > ')} {/* Replace dots with arrows for better readability */}
                    </option>
                  ))}
                </select>
              </div>
              {selectedCategory && (
                <div className="selected-category-info">
                  <span className="category-tag">Selected: {selectedCategory.replace(/\./g, ' > ')}</span>
                  <button 
                    className="clear-category-btn"
                    onClick={() => setSelectedCategory('')}
                  >
                    Clear
                  </button>
                </div>
              )}
            </div>
            <p className="category-help-text">
              Select a category to get recommendations based only on items you've interacted with in that category.
              Categories are hierarchical - selecting "electronics" includes all electronics subcategories.
            </p>
          </div>
        )}

        {/* Recommendation Controls */}
        <div className="recommendation-controls">
          <h2>Recommendation Settings</h2>
          
          <div className="controls-row">
            <div className="form-group">
              <label htmlFor="recType">Recommendation Type:</label>
              <select
                id="recType"
                value={recommendationType}
                onChange={(e) => setRecommendationType(e.target.value)}
              >
                <option value="hybrid">Hybrid</option>
                <option value="enhanced">🎯 Enhanced Hybrid (Category-Aware)</option>
                <option value="category_focused">🎯 Category Focused (80% Match)</option>
                <option value="collaborative">Collaborative Filtering</option>
                <option value="content">Content-Based</option>
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="numRecs">Number of Recommendations:</label>
              <select
                id="numRecs"
                value={numRecommendations}
                onChange={(e) => setNumRecommendations(parseInt(e.target.value))}
              >
                <option value="10">10</option>
                <option value="20">20</option>
                <option value="50">50</option>
                <option value="100">100 (Top Items)</option>
                <option value="200">200 (Extended)</option>
              </select>
            </div>
            
            {(recommendationType === 'hybrid' || recommendationType === 'enhanced') && (
              <div className="form-group">
                <label htmlFor="collabWeight">Collaborative Weight:</label>
                <input
                  type="range"
                  id="collabWeight"
                  min="0"
                  max="1"
                  step="0.1"
                  value={collaborativeWeight}
                  onChange={(e) => setCollaborativeWeight(parseFloat(e.target.value))}
                />
                <span>{collaborativeWeight}</span>
              </div>
            )}
            
            <button
              className="btn btn-primary"
              onClick={getRecommendations}
              disabled={loading || (recommendationType === 'content' && userProfile.interaction_history.length === 0)}
            >
              {loading ? 'Loading...' : 'Get Recommendations'}
            </button>
          </div>
          
          {recommendationType === 'content' && userProfile.interaction_history.length === 0 && (
            <p style={{color: '#dc3545', marginTop: '10px', fontSize: '14px'}}>
              Content-based recommendations require interaction history. Please select an interaction pattern above.
            </p>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Recommendations Display */}
        {recommendations.length > 0 && (
          <div className="recommendations">
            <h2>Top {recommendations.length} Recommendations ({recommendationType})</h2>
            
            <div className="stats">
              <strong>User Profile:</strong> {userProfile.age}yr {userProfile.gender}, 
              ${userProfile.income.toLocaleString()} income
              {selectedCategory && (
                <span> | <strong>Category Filter:</strong> <span className="category-filter-display">{selectedCategory.replace(/\./g, ' > ')}</span></span>
              )}
              {useRealUsers && selectedRealUser ? (
                <span> | <strong>Real User {selectedRealUser.user_id}:</strong> {selectedRealUser.interaction_pattern} - 
                {selectedRealUser.interaction_stats.total_interactions} total interactions 
                ({selectedRealUser.interaction_stats.views} views, {selectedRealUser.interaction_stats.cart_adds} carts, {selectedRealUser.interaction_stats.purchases} purchases)
                </span>
              ) : selectedBehavioralPattern ? (
                <span> | <strong>Behavioral Pattern:</strong> {selectedBehavioralPattern.pattern} - 
                {selectedBehavioralPattern.stats.total_interactions} total interactions 
                ({selectedBehavioralPattern.stats.views} views, {selectedBehavioralPattern.stats.cart_adds} carts, {selectedBehavioralPattern.stats.purchases} purchases)
                </span>
              ) : (
                <span> | <strong>Synthetic:</strong> {interactions.length} total interactions ({counts.views || 0} views, {counts.carts || 0} carts, {counts.purchases || 0} purchases)</span>
              )}
            </div>
            
            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div className="pagination-info">
                <p>Showing {startIndex + 1}-{Math.min(endIndex, recommendations.length)} of {recommendations.length} recommendations</p>
                <div className="pagination-controls">
                  <button 
                    onClick={() => goToPage(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="pagination-btn"
                  >
                    ← Previous
                  </button>
                  
                  <div className="page-numbers">
                    {Array.from({length: Math.min(totalPages, 10)}, (_, i) => {
                      const page = i + 1;
                      return (
                        <button
                          key={page}
                          onClick={() => goToPage(page)}
                          className={`page-number ${currentPage === page ? 'active' : ''}`}
                        >
                          {page}
                        </button>
                      );
                    })}
                    {totalPages > 10 && <span className="pagination-ellipsis">...</span>}
                  </div>
                  
                  <button 
                    onClick={() => goToPage(currentPage + 1)}
                    disabled={currentPage === totalPages}
                    className="pagination-btn"
                  >
                    Next →
                  </button>
                </div>
              </div>
            )}
            
            <div className="recommendations-grid">
              {currentRecommendations.map((rec, index) => (
                <div 
                  key={rec.item_id} 
                  className="recommendation-card clickable-card"
                  onClick={() => handleItemClick(rec.item_id, rec.item_info)}
                  title="Click to find similar items"
                >
                  <div className="card-header">
                    <span className="item-id">#{startIndex + index + 1} Item {rec.item_id}</span>
                    <span className="score">{rec.score.toFixed(4)}</span>
                  </div>
                  
                  <div className="item-details">
                    <p className="brand">{rec.item_info.brand}</p>
                    <p className="price">${rec.item_info.price.toFixed(2)}</p>
                    <p className="category">{rec.item_info.category_code}</p>
                  </div>
                  
                  <div className="click-hint">
                    <small>🔍 Click for similar items</small>
                  </div>
                </div>
              ))}
            </div>
            
            {/* Bottom Pagination */}
            {totalPages > 1 && (
              <div className="pagination-controls bottom-pagination">
                <button 
                  onClick={() => goToPage(currentPage - 1)}
                  disabled={currentPage === 1}
                  className="pagination-btn"
                >
                  ← Previous
                </button>
                <span className="page-indicator">Page {currentPage} of {totalPages}</span>
                <button 
                  onClick={() => goToPage(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className="pagination-btn"
                >
                  Next →
                </button>
              </div>
            )}
          </div>
        )}

        {loading && (
          <div className="loading">
            <h3>Generating recommendations...</h3>
            <p>Analyzing your interaction patterns and preferences...</p>
          </div>
        )}

        {/* Performance Monitoring Widget */}
        {requestHistory.length > 0 && (
          <div className="performance-widget">
            <div className="performance-header" onClick={() => setShowPerformanceDetails(!showPerformanceDetails)}>
              <div className="performance-summary">
                <div className="perf-icon">⚡</div>
                <div className="perf-stats">
                  <div className="perf-time">{Math.round(requestHistory[0].duration)}ms</div>
                  <div className="perf-label">Last Request</div>
                </div>
              </div>
              <div className="perf-expand-btn">
                {showPerformanceDetails ? '▼' : '▶'}
              </div>
            </div>
            
            {showPerformanceDetails && (
              <div className="performance-details">
                <div className="performance-title">Request Performance History</div>
                <div className="performance-list">
                  {requestHistory.map((record) => (
                    <div key={record.id} className="performance-item">
                      <div className="perf-item-header">
                        <div className="perf-item-time">
                          <strong>{Math.round(record.duration)}ms</strong>
                        </div>
                        <div className="perf-item-timestamp">
                          {record.timestamp.toLocaleTimeString()}
                        </div>
                        <div className="perf-item-type">
                          {record.requestDetails.recommendationType}
                        </div>
                      </div>
                      <div className="perf-item-details">
                        <div className="perf-detail-row">
                          <span>Type:</span> {record.requestDetails.recommendationType}
                          {record.requestDetails.selectedCategory !== 'None' && (
                            <span> | Category: {record.requestDetails.selectedCategory}</span>
                          )}
                        </div>
                        <div className="perf-detail-row">
                          <span>User:</span> {record.requestDetails.userAge}yr {record.requestDetails.userGender}
                          <span> | History: {record.requestDetails.interactionHistorySize} items</span>
                        </div>
                        <div className="perf-detail-row">
                          <span>Request:</span> {record.requestDetails.numRecommendations} items
                          <span> | Response: {record.responseDetails.recommendationCount} items</span>
                          {record.responseDetails.error && <span className="error-indicator"> | ERROR</span>}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="performance-stats-summary">
                  <div className="stat-item">
                    <strong>Avg:</strong> {Math.round(requestHistory.reduce((sum, r) => sum + r.duration, 0) / requestHistory.length)}ms
                  </div>
                  <div className="stat-item">
                    <strong>Min:</strong> {Math.round(Math.min(...requestHistory.map(r => r.duration)))}ms
                  </div>
                  <div className="stat-item">
                    <strong>Max:</strong> {Math.round(Math.max(...requestHistory.map(r => r.duration)))}ms
                  </div>
                  <div className="stat-item">
                    <strong>Total Requests:</strong> {requestHistory.length}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Similar Items Modal */}
        {showSimilarItems && selectedItemForSimilarity && (
          <div className="similar-items-overlay">
            <div className="similar-items-modal">
              <div className="modal-header">
                <h3>Items Similar to #{selectedItemForSimilarity.id}</h3>
                <button className="close-btn" onClick={closeSimilarItems}>✕</button>
              </div>

              <div className="original-item">
                <h4>Original Item</h4>
                <div className="original-item-card">
                  <div className="item-header">
                    <span className="item-id">Item {selectedItemForSimilarity.id}</span>
                  </div>
                  <div className="item-details">
                    <p className="brand"><strong>{selectedItemForSimilarity.info.brand}</strong></p>
                    <p className="price">${selectedItemForSimilarity.info.price.toFixed(2)}</p>
                    <p className="category">{selectedItemForSimilarity.info.category_code}</p>
                  </div>
                </div>
              </div>

              <div className="similar-items-section">
                <h4>Similar Items {loadingSimilarItems ? '(Loading...)' : `(${similarItems.length})`}</h4>
                
                {loadingSimilarItems ? (
                  <div className="loading-similar-items">
                    <div className="loading-spinner"></div>
                    <p>Finding similar items using ANN search...</p>
                  </div>
                ) : (
                  <div className="similar-items-grid">
                    {similarItems.map((item, index) => (
                      <div key={item.item_id} className="similar-item-card">
                        <div className="similarity-badge">
                          <span className="similarity-score">{(item.score * 100).toFixed(1)}%</span>
                          <small>similarity</small>
                        </div>
                        
                        <div className="item-header">
                          <span className="item-id">#{index + 1} Item {item.item_id}</span>
                        </div>
                        
                        <div className="item-details">
                          <p className="brand"><strong>{item.item_info.brand}</strong></p>
                          <p className="price">${item.item_info.price.toFixed(2)}</p>
                          <p className="category">{item.item_info.category_code}</p>
                        </div>
                        
                        <div className="similarity-bar">
                          <div 
                            className="similarity-fill" 
                            style={{width: `${item.score * 100}%`}}
                          ></div>
                        </div>
                      </div>
                    ))}
                    
                    {similarItems.length === 0 && !loadingSimilarItems && (
                      <div className="no-similar-items">
                        <p>No similar items found</p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="modal-footer">
                <small>Similarity scores computed using FAISS ANN search with cosine similarity</small>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;