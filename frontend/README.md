# YieldWise React Frontend

🌾 **Smart Crop Advisor UI** - Modern React frontend for the YieldWise agricultural intelligence platform.

## 🚀 Features

- **🏠 Home**: Landing page with platform overview and getting started guide
- **📊 Yield Predictor**: AI-powered crop yield forecasting with interactive forms
- **📈 Dashboard**: Comprehensive overview of all agricultural insights
- **💰 Financial Calculator**: Cost-benefit analysis and ROI calculations
- **🔬 Disease Detection**: Computer vision-based plant disease identification

## 🛠️ Tech Stack

- **Framework**: React 18.2.0
- **Routing**: React Router DOM 6.3.0
- **HTTP Client**: Axios 1.4.0
- **Charts**: Chart.js 4.3.0 + React-ChartJS-2 5.2.0
- **Maps**: Mapbox GL 2.15.0 + React-Map-GL 7.1.0
- **Build Tool**: Create React App
- **Testing**: React Testing Library

## 📁 Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── favicon.ico
├── src/
│   ├── components/
│   │   ├── Navbar.js          # Navigation component with routing
│   │   ├── Navbar.css         # Navigation styles
│   │   ├── Footer.js          # Footer component
│   │   └── LoadingSpinner.js  # Reusable loading component
│   ├── pages/
│   │   ├── Home.js            # Landing page component
│   │   ├── YieldPredictor.js  # Yield prediction interface
│   │   ├── Dashboard.js       # Analytics dashboard
│   │   ├── FinancialCalculator.js  # Financial analysis tool
│   │   └── DiseaseDetection.js     # Disease detection interface
│   ├── styles/
│   │   ├── App.css           # Global application styles
│   │   ├── Home.css          # Home page styles
│   │   ├── YieldPredictor.css # Yield predictor styles
│   │   ├── Dashboard.css     # Dashboard styles
│   │   ├── FinancialCalculator.css # Financial calculator styles
│   │   └── DiseaseDetection.css    # Disease detection styles
│   ├── services/
│   │   ├── api.js            # API service layer
│   │   ├── yieldService.js   # Yield prediction API calls
│   │   ├── weatherService.js # Weather data integration
│   │   ├── financialService.js # Financial calculations API
│   │   └── diseaseService.js # Disease detection API
│   ├── utils/
│   │   ├── constants.js      # Application constants
│   │   ├── helpers.js        # Utility functions
│   │   └── validators.js     # Form validation helpers
│   ├── App.js                # Main application component
│   ├── App.css              # Main application styles
│   ├── index.js             # Application entry point
│   └── index.css            # Base styles
├── package.json             # Dependencies and scripts
└── README.md               # This file
```

## 🏃‍♂️ Getting Started

### Prerequisites

- Node.js 16+ and npm
- Backend API server running on http://localhost:5000

### Installation

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   Create a `.env` file in the frontend directory:
   ```env
   REACT_APP_API_URL=http://localhost:5000
   REACT_APP_MAPBOX_TOKEN=your_mapbox_token_here
   ```

4. **Start development server**
   ```bash
   npm start
   ```

5. **Open your browser**
   Navigate to http://localhost:3000

## 📱 Pages Overview

### 🏠 Home Page (`/`)
- Welcome section with platform introduction
- Feature highlights and benefits
- Quick navigation to main tools
- Getting started guide

### 📊 Yield Predictor (`/yield-predictor`)
- Interactive form for crop details input
- Location selection with Mapbox integration
- Real-time weather data integration
- AI-powered yield predictions with confidence intervals
- Downloadable reports

### 📈 Dashboard (`/dashboard`)
- Overview of recent predictions
- Weather insights and alerts
- Financial summaries
- Disease detection history
- Quick access to all tools

### 💰 Financial Calculator (`/financial-calculator`)
- Comprehensive cost breakdown forms
- ROI calculations and projections
- Interactive charts and visualizations
- Market price insights
- Export financial reports

### 🔬 Disease Detection (`/disease-detection`)
- Image upload interface
- Real-time disease analysis
- Treatment recommendations
- Disease prevention tips
- Results history and tracking

## 🔄 API Integration

The frontend communicates with the Flask backend via RESTful APIs:

- **POST /predict** - Crop yield predictions
- **POST /weather** - Weather data fetching
- **POST /calculate-finance** - Financial analysis
- **POST /predict-disease** - Disease detection
- **POST /geocode** - Location services

## 🎨 Styling Approach

- **CSS Modules** for component-specific styles
- **Responsive Design** with mobile-first approach
- **Modern UI** with clean, agricultural-themed design
- **Accessibility** features for all users
- **Dark/Light Mode** toggle (future enhancement)

## 🧪 Testing

Run the test suite:
```bash
npm test
```

Run tests with coverage:
```bash
npm test -- --coverage
```

## 🏗️ Building for Production

1. **Create production build**
   ```bash
   npm run build
   ```

2. **Serve static files**
   The `build/` folder contains optimized static files ready for deployment.

## 🚀 Deployment Options

### Vercel (Recommended)
```bash
npm install -g vercel
vercel --prod
```

### Netlify
```bash
npm run build
# Upload build/ folder to Netlify
```

### Docker
```dockerfile
FROM node:16-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 🔧 Development Guidelines

### Component Structure
- Use functional components with hooks
- Implement proper error boundaries
- Follow React best practices for performance
- Use TypeScript for type safety (future enhancement)

### State Management
- Local component state with useState
- Context API for global state (user preferences)
- Consider Redux Toolkit for complex state (future)

### Code Style
- ESLint and Prettier configuration
- Consistent naming conventions
- Comprehensive prop validation
- JSDoc comments for complex functions

## 🌟 Future Enhancements

- [ ] **TypeScript Migration** - Add type safety
- [ ] **PWA Features** - Offline functionality
- [ ] **Real-time Updates** - WebSocket integration
- [ ] **Advanced Charts** - D3.js integration
- [ ] **Mobile App** - React Native version
- [ ] **Internationalization** - Multi-language support
- [ ] **Advanced Animations** - Framer Motion integration
- [ ] **Voice Commands** - Speech recognition

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

1. **Port 3000 already in use**
   ```bash
   npx kill-port 3000
   npm start
   ```

2. **API connection errors**
   - Ensure backend is running on port 5000
   - Check CORS configuration
   - Verify API endpoints

3. **Build failures**
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   npm run build
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- React team for the amazing framework
- Create React App for the build tooling
- Chart.js community for visualization tools
- Mapbox for mapping services
- All contributors to the agricultural data science community

---

**Built with ❤️ for farmers and agricultural professionals worldwide** 🌾
