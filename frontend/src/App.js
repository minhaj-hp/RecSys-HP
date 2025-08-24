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

  // Load sample items and real users on component mount
  useEffect(() => {
    fetchSampleItems();
    fetchRealUsers();
    fetchDatasetSummary();
  }, []);

  const fetchSampleItems = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/items?limit=50`);
      setSampleItems(response.data.items || []);
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
    if (sampleItems.length === 0) return;

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
    if (!selectedRealUser || !userInteractionDetails) return {};
    
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

  const categoryPercentages = getCategoryPercentages();
  const recommendationCategoryPercentages = getRecommendationCategoryPercentages();

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
    
    try {
      const requestData = {
        user_profile: userProfile,
        num_recommendations: numRecommendations,
        recommendation_type: recommendationType,
        collaborative_weight: collaborativeWeight
      };
      
      const response = await axios.post(`${API_BASE_URL}/recommendations`, requestData);
      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError(error.response?.data?.detail || 'Failed to fetch recommendations');
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
              <h2>Synthetic Interaction Patterns</h2>
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
              {useRealUsers && selectedRealUser ? (
                <span> | <strong>Real User {selectedRealUser.user_id}:</strong> {selectedRealUser.interaction_pattern} - 
                {selectedRealUser.interaction_stats.total_interactions} total interactions 
                ({selectedRealUser.interaction_stats.views} views, {selectedRealUser.interaction_stats.cart_adds} carts, {selectedRealUser.interaction_stats.purchases} purchases)
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
                <div key={rec.item_id} className="recommendation-card">
                  <div className="card-header">
                    <span className="item-id">#{startIndex + index + 1} Item {rec.item_id}</span>
                    <span className="score">{rec.score.toFixed(4)}</span>
                  </div>
                  
                  <div className="item-details">
                    <p className="brand">{rec.item_info.brand}</p>
                    <p className="price">${rec.item_info.price.toFixed(2)}</p>
                    <p className="category">{rec.item_info.category_code}</p>
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
      </div>
    </div>
  );
}

export default App;