// src/api/apiClient.js - Fixed version with pagination and memory management
class APIError extends Error {
  constructor(message, status, response) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.response = response;
  }
}

class APIClient {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
    this.requestCache = new Map();
    this.maxCacheSize = 50;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    // Check cache for GET requests
    if (!options.method || options.method === 'GET') {
      const cached = this.requestCache.get(url);
      if (cached && Date.now() - cached.timestamp < 60000) { // 1 min cache
        if (process.env.NODE_ENV === 'development') {
          console.log(`📦 Cached: ${url}`);
        }
        return cached.data;
      }
    }

    const config = {
      headers: { ...this.defaultHeaders, ...options.headers },
      ...options,
    };

    // Add request logging for development
    if (process.env.NODE_ENV === 'development') {
      console.log(`🌐 API Request: ${config.method || 'GET'} ${url}`, {
        body: config.body ? JSON.parse(config.body) : undefined,
      });
    }

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new APIError(
          data.detail || `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          data
        );
      }

      // Cache GET requests
      if (!options.method || options.method === 'GET') {
        this.requestCache.set(url, { data, timestamp: Date.now() });
        // Evict old entries if cache is too large
        if (this.requestCache.size > this.maxCacheSize) {
          const firstKey = this.requestCache.keys().next().value;
          this.requestCache.delete(firstKey);
        }
      }

      // Log successful responses in development
      if (process.env.NODE_ENV === 'development') {
        console.log(`✅ API Response: ${config.method || 'GET'} ${url}`, data);
      }

      return data;
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      
      // Handle network errors
      console.error(`❌ API Network Error: ${config.method || 'GET'} ${url}`, error);
      throw new APIError(
        'Network error - check if backend is running on http://localhost:8000',
        0,
        null
      );
    }
  }

  // GET request helper
  async get(endpoint, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const url = queryString ? `${endpoint}?${queryString}` : endpoint;
    return this.request(url);
  }

  // POST request helper
  async post(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // PUT request helper
  async put(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  // DELETE request helper
  async delete(endpoint) {
    return this.request(endpoint, {
      method: 'DELETE',
    });
  }

  /**
   * Fetch all tracks with automatic pagination
   * This prevents ERR_INSUFFICIENT_RESOURCES by batching requests
   */
  async getAllTracksPaginated(batchSize = 100) {
    let allTracks = [];
    let offset = 0;
    let hasMore = true;

    console.log('🎵 Starting paginated track fetch...');

    try {
      while (hasMore) {
        const response = await this.get('/tracks', {
          limit: batchSize,
          offset: offset
        });

        if (response.tracks && response.tracks.length > 0) {
          allTracks = [...allTracks, ...response.tracks];
          offset += batchSize;
          hasMore = response.has_more || false;
          
          console.log(`📥 Loaded ${allTracks.length}/${response.total} tracks`);
          
          // Add small delay between requests to prevent overwhelming the server
          if (hasMore) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }
        } else {
          hasMore = false;
        }

        // Safety limit: don't load more than 2000 tracks at once
        if (allTracks.length >= 2000) {
          console.warn('⚠️ Reached 2000 track limit, stopping pagination');
          hasMore = false;
        }
      }

      console.log(`✅ Loaded ${allTracks.length} total tracks`);
      return allTracks;
    } catch (error) {
      console.error('❌ Pagination failed:', error);
      throw error;
    }
  }

  /**
   * Clear request cache
   */
  clearCache() {
    this.requestCache.clear();
    console.log('🗑️ Request cache cleared');
  }
}

// Create singleton instance
export const apiClient = new APIClient();
export { APIError };