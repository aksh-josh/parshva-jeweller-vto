import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Auth() {
  const navigate = useNavigate();

  // Component State
  const [activeTab, setActiveTab] = useState('signin'); // 'signin' | 'signup'
  const [step, setStep] = useState(1); // 1 = Phone/Details, 2 = OTP
  const [phone, setPhone] = useState('');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ text: '', type: '' });
  const [devOtp, setDevOtp] = useState(null);
  
  const [resendCooldown, setResendCooldown] = useState(0);

  // Refs for OTP inputs auto-focusing
  const otpRefs = useRef([]);

  // Reset state when switching tabs
  const switchTab = (tab) => {
    setActiveTab(tab);
    setStep(1);
    setMessage({ text: '', type: '' });
    setDevOtp(null);
    setOtp(['', '', '', '', '', '']);
  };

  const handlePhoneChange = (e) => {
    const val = e.target.value.replace(/\D/g, ''); // Only allow digits
    setPhone(val);
  };

  const handleSignIn = async (e) => {
    e.preventDefault();
    if (phone.length !== 10) {
      setMessage({ text: 'Please enter a valid 10-digit phone number.', type: 'error' });
      return;
    }
    
    setLoading(true);
    try {
      // IN PRODUCTION: Point this to your actual Flask backend container
      // const res = await fetch('/api/auth/login', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ phone })
      // });
      // const data = await res.json();
      
      // MOCK RESPONSE FOR TESTING:
      const data = { success: true, message: 'OTP Sent successfully', dev_otp: '123456' };

      if (data.success) {
        setStep(2);
        setMessage({ text: data.message, type: 'success' });
        if (data.dev_otp) setDevOtp(data.dev_otp);
        setTimeout(() => otpRefs.current[0]?.focus(), 100);
      } else {
        setMessage({ text: data.message || 'Error occurred', type: 'error' });
      }
    } catch (err) {
      setMessage({ text: 'Network error. Please try again.', type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleSignUp = async (e) => {
    e.preventDefault();
    if (name.length < 2) {
      setMessage({ text: 'Please enter your full name.', type: 'error' });
      return;
    }
    if (phone.length !== 10) {
      setMessage({ text: 'Please enter a valid 10-digit phone number.', type: 'error' });
      return;
    }

    setLoading(true);
    try {
      // MOCK RESPONSE
      const data = { success: true, message: 'OTP Sent successfully', dev_otp: '654321' };

      if (data.success) {
        setStep(2);
        setMessage({ text: data.message, type: 'success' });
        if (data.dev_otp) setDevOtp(data.dev_otp);
        setTimeout(() => otpRefs.current[0]?.focus(), 100);
      } else {
        setMessage({ text: data.message || 'Error occurred', type: 'error' });
      }
    } catch (err) {
      setMessage({ text: 'Network error. Please try again.', type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const verifyOTP = async (e) => {
    e.preventDefault();
    const otpString = otp.join('');
    if (otpString.length !== 6) {
      setMessage({ text: 'Please enter the full 6-digit code.', type: 'error' });
      return;
    }

    setLoading(true);
    try {
      // MOCK RESPONSE
      const data = { success: true, message: 'Verified Successfully!' };

      if (data.success) {
        setMessage({ text: data.message, type: 'success' });
        // Redirect to homepage after successful login
        setTimeout(() => navigate('/'), 800); 
      } else {
        setMessage({ text: data.message || 'Invalid OTP', type: 'error' });
        setOtp(['', '', '', '', '', '']);
        otpRefs.current[0]?.focus();
      }
    } catch (err) {
      setMessage({ text: 'Network error. Please try again.', type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const resendOTP = async () => {
    if (resendCooldown > 0) return;
    
    try {
      // MOCK RESPONSE
      const data = { success: true, message: 'New OTP Sent', dev_otp: '999999' };
      
      setMessage({ text: data.message, type: 'success' });
      if (data.dev_otp) setDevOtp(data.dev_otp);
      
      // Start 30s cooldown timer
      setResendCooldown(30);
    } catch (err) {
      setMessage({ text: 'Failed to resend. Try again.', type: 'error' });
    }
  };

  // Cooldown Timer Effect
  useEffect(() => {
    let timer;
    if (resendCooldown > 0) {
      timer = setInterval(() => {
        setResendCooldown(prev => prev - 1);
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [resendCooldown]);

  // --- OTP Input Logic ---
  const handleOtpChange = (index, value) => {
    if (isNaN(value)) return;
    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);

    // Auto-focus next input
    if (value && index < 5) {
      otpRefs.current[index + 1].focus();
    }
  };

  const handleOtpKeyDown = (index, e) => {
    if (e.key === 'Backspace' && !otp[index] && index > 0) {
      otpRefs.current[index - 1].focus();
    }
  };

  const handleOtpPaste = (e) => {
    e.preventDefault();
    const pastedData = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    if (pastedData.length === 6) {
      setOtp(pastedData.split(''));
      otpRefs.current[5].focus();
    }
  };

  return (
    <section className="min-h-[85vh] flex items-center justify-center py-16 px-4 bg-gradient-to-br from-yellow-50 via-white to-yellow-50">
      <div className="w-full max-w-md mt-10">
        
        {/* Logo */}
        <div className="text-center mb-8">
          <h1 className="text-4xl brand-font text-gray-900">Parshva Jewellers</h1>
          <p className="text-gray-500 mt-2">Welcome back to timeless elegance</p>
        </div>

        {/* Auth Card */}
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
          
          {/* Tabs */}
          <div className="flex border-b border-gray-100">
            <button
              onClick={() => switchTab('signin')}
              className={`flex-1 py-4 text-sm font-semibold tracking-wide uppercase transition-all duration-300 border-b-2 ${activeTab === 'signin' ? 'text-yellow-700 border-yellow-700' : 'text-gray-400 border-transparent hover:text-gray-600'}`}
            >
              Sign In
            </button>
            <button
              onClick={() => switchTab('signup')}
              className={`flex-1 py-4 text-sm font-semibold tracking-wide uppercase transition-all duration-300 border-b-2 ${activeTab === 'signup' ? 'text-yellow-700 border-yellow-700' : 'text-gray-400 border-transparent hover:text-gray-600'}`}
            >
              Create Account
            </button>
          </div>

          <div className="p-8">
            {/* Step 1: Phone / Details */}
            {step === 1 && (
              <form onSubmit={activeTab === 'signin' ? handleSignIn : handleSignUp}>
                <p className="text-gray-600 text-sm mb-6">
                  {activeTab === 'signin' 
                    ? "Enter your phone number to receive a verification code." 
                    : "Create your account to start shopping."}
                </p>

                {activeTab === 'signup' && (
                  <div className="mb-4">
                    <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Full Name</label>
                    <input 
                      type="text" 
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Enter your name" 
                      className="w-full px-4 py-3 border border-gray-200 rounded-lg outline-none focus:border-yellow-600 focus:ring-1 focus:ring-yellow-600 transition text-gray-800 placeholder-gray-300"
                    />
                  </div>
                )}

                <div className="mb-4">
                  <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Phone Number</label>
                  <div className="flex border border-gray-200 rounded-lg overflow-hidden focus-within:border-yellow-600 focus-within:ring-1 focus-within:ring-yellow-600 transition">
                    <span className="bg-gray-50 px-3 flex items-center text-gray-500 text-sm border-r border-gray-200">+91</span>
                    <input 
                      type="tel" 
                      maxLength="10" 
                      value={phone}
                      onChange={handlePhoneChange}
                      placeholder="Enter 10-digit number" 
                      className="flex-1 px-4 py-3 outline-none text-gray-800 placeholder-gray-300"
                    />
                  </div>
                </div>

                {activeTab === 'signup' && (
                  <div className="mb-5">
                    <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Email <span className="text-gray-300">(optional)</span></label>
                    <input 
                      type="email" 
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="your@email.com" 
                      className="w-full px-4 py-3 border border-gray-200 rounded-lg outline-none focus:border-yellow-600 focus:ring-1 focus:ring-yellow-600 transition text-gray-800 placeholder-gray-300"
                    />
                  </div>
                )}

                <button 
                  type="submit" 
                  disabled={loading}
                  className={`w-full bg-yellow-700 hover:bg-yellow-800 text-white py-3 rounded-lg font-semibold transition-all duration-300 shadow-md hover:shadow-lg ${loading ? 'opacity-70 cursor-not-allowed' : ''}`}
                >
                  {loading ? 'Please wait...' : (activeTab === 'signin' ? 'Send OTP' : 'Send Verification Code')}
                </button>
              </form>
            )}

            {/* Step 2: OTP Verification */}
            {step === 2 && (
              <form onSubmit={verifyOTP}>
                <p className="text-gray-600 text-sm mb-2">We sent a code to</p>
                <p className="text-gray-900 font-semibold mb-2">+91 {phone}</p>
                <p className="text-xs text-green-600 mb-4">✓ A fresh code has been sent. Previous codes are now invalid.</p>

                <div className="mb-5">
                  <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Verification Code</label>
                  <div className="flex gap-2 justify-center" onPaste={handleOtpPaste}>
                    {otp.map((digit, index) => (
                      <input
                        key={index}
                        type="text"
                        maxLength="1"
                        ref={el => otpRefs.current[index] = el}
                        value={digit}
                        onChange={(e) => handleOtpChange(index, e.target.value)}
                        onKeyDown={(e) => handleOtpKeyDown(index, e)}
                        className="w-12 h-14 text-center text-xl font-bold border border-gray-200 rounded-lg focus:border-yellow-600 focus:ring-1 focus:ring-yellow-600 outline-none transition"
                      />
                    ))}
                  </div>
                </div>

                <button 
                  type="submit" 
                  disabled={loading}
                  className={`w-full bg-yellow-700 hover:bg-yellow-800 text-white py-3 rounded-lg font-semibold transition-all duration-300 shadow-md hover:shadow-lg ${loading ? 'opacity-70 cursor-not-allowed' : ''}`}
                >
                  {loading ? 'Verifying...' : (activeTab === 'signin' ? 'Verify & Sign In' : 'Verify & Create Account')}
                </button>

                <div className="flex justify-between items-center mt-4">
                  <button type="button" onClick={() => setStep(1)} className="text-sm text-gray-500 hover:text-gray-700">
                    ← Change number
                  </button>
                  <button 
                    type="button" 
                    onClick={resendOTP}
                    disabled={resendCooldown > 0}
                    className={`text-sm font-semibold transition-colors ${resendCooldown > 0 ? 'text-gray-400 cursor-not-allowed' : 'text-yellow-700 hover:text-yellow-800'}`}
                  >
                    {resendCooldown > 0 ? `Resend in ${resendCooldown}s` : 'Resend OTP'}
                  </button>
                </div>
              </form>
            )}

            {/* Status Message */}
            {message.text && (
              <div className={`mt-4 text-center text-sm ${message.type === 'error' ? 'text-red-600' : 'text-green-600'}`}>
                {message.text}
              </div>
            )}

            {/* Dev OTP Display */}
            {devOtp && (
              <div className="mt-3 text-center text-xs text-gray-400">
                Dev OTP: <span className="font-mono font-bold text-yellow-700">{devOtp}</span>
              </div>
            )}
          </div>
        </div>

        <p className="text-center text-xs text-gray-400 mt-6">
          By continuing, you agree to our Terms of Service and Privacy Policy.
        </p>
      </div>
    </section>
  );
}