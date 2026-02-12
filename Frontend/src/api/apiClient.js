// src/api/apiClient.js
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
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
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
      throw new APIError('Network error - please check your connection', 0, null);
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
}

// Create singleton instance
export const apiClient = new APIClient();
export { APIError };
