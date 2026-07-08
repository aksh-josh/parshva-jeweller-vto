# Use the official Node image (Debian-based)
FROM node:20-slim

# Set the working directory inside the container
WORKDIR /app

# Copy ONLY package.json and install Node modules fresh
COPY package.json ./
RUN npm install

# Copy all the React source code into the container
COPY . .

# Expose the port Vite runs on
EXPOSE 5173

# Run the React development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
