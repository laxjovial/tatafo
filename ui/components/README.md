# 1. Create a new Vite React project
npm create vite@latest my-intelli-agent-frontend -- --template react

# 2. Navigate into the new project directory
cd my-intelli-agent-frontend

# 3. Install Tailwind CSS (if not already set up by Vite template)
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# 4. Configure Tailwind in tailwind.config.js and index.css (refer to Tailwind docs for exact steps)
#    You'll need to add your React component paths to content array in tailwind.config.js
#    And import Tailwind into your index.css

# 5. Install Firebase and other dependencies
npm install firebase recharts lucide-react

# 6. Replace src/App.jsx with the content I provided
#    Also, ensure you have src/LoginPage.jsx, src/RegisterPage.jsx, etc. if they are separate files.
#    (In my previous response, I put all components into a single App.jsx for simplicity in the Canvas,
#     but in a real project, you'd split them into separate files).

# 7. Run the development server
npm run dev
