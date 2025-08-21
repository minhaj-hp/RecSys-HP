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
  const [numRecommendations, setNumRecommendations] = useState(10);
  const [collaborativeWeight, setCollaborativeWeight] = useState(0.7);
  
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [sampleItems, setSampleItems] = useState([]);
  const [interactions, setInteractions] = useState([]);
  const [expandedInteraction, setExpandedInteraction] = useState(null);
  const [selectedPattern, setSelectedPattern] = useState(null);

  // Load sample items on component mount
  useEffect(() => {
    fetchSampleItems();
  }, []);

  const fetchSampleItems = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/items?limit=50`);
      setSampleItems(response.data.items || []);
    } catch (error) {
      console.error('Error fetching sample items:', error);
    }
  };

  const handleProfileChange = (field, value) => {
    setUserProfile(prev => ({
      ...prev,
      [field]: field === 'age' || field === 'income' ? parseInt(value) || 0 : value
    }));
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

  const getRecommendations = async () => {
    setLoading(true);
    setError(null);
    
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

  const getInteractionCounts = () => {
    const counts = { views: 0, carts: 0, purchases: 0 };
    interactions.forEach(interaction => {
      counts[interaction.type + 's'] = (counts[interaction.type + 's'] || 0) + 1;
    });
    return counts;
  };

  const counts = getInteractionCounts();

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>Two-Tower Recommendation System Demo</h1>
          <p>Configure user demographics and realistic interaction patterns to get personalized recommendations</p>
        </header>

        {/* User Profile Form */}
        <div className="user-profile-form">
          <h2>User Demographics</h2>
          
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
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="gender">Gender:</label>
              <select
                id="gender"
                value={userProfile.gender}
                onChange={(e) => handleProfileChange('gender', e.target.value)}
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
              />
            </div>
          </div>
        </div>

        {/* Interaction Patterns */}
        <div className="interaction-patterns">
          <h2>Interaction Patterns</h2>
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
                        <strong>{interaction.brand}</strong> - ${interaction.price}
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
                <option value="5">5</option>
                <option value="10">10</option>
                <option value="15">15</option>
                <option value="20">20</option>
              </select>
            </div>
            
            {recommendationType === 'hybrid' && (
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
            <h2>Recommendations ({recommendationType})</h2>
            
            <div className="stats">
              <strong>User Profile:</strong> {userProfile.age}yr {userProfile.gender}, 
              ${userProfile.income.toLocaleString()} income, 
              {interactions.length} total interactions ({counts.views || 0} views, {counts.carts || 0} carts, {counts.purchases || 0} purchases)
            </div>
            
            <div className="recommendations-grid">
              {recommendations.map((rec, index) => (
                <div key={rec.item_id} className="recommendation-card">
                  <div className="card-header">
                    <span className="item-id">#{index + 1} Item {rec.item_id}</span>
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