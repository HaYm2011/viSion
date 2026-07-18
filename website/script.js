document.addEventListener('DOMContentLoaded', () => {
  // Scroll to top on load
  window.scrollTo(0, 0);

  // Loader Animation
  const loader = document.getElementById('loader');
  const loaderLogo = loader.querySelector('.logo');
  const heroTitle = document.getElementById('hero-title');
  const heroContent = document.querySelector('.hero-content');

  setTimeout(() => {
    // Grow the logo
    loaderLogo.style.fontSize = '6rem';

    setTimeout(() => {
      const targetRect = heroTitle.getBoundingClientRect();
      const loaderRect = loaderLogo.getBoundingClientRect();

      const scale = targetRect.width / loaderRect.width;
      const translateX = targetRect.left - loaderRect.left;
      const translateY = targetRect.top - loaderRect.top;

      loaderLogo.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;

      setTimeout(() => {
        heroTitle.style.opacity = '1';
        loader.style.display = 'none';
        heroContent.classList.add('visible');
      }, 1000); // This should match the transition duration in CSS
    }, 1000);
  }, 1000);

  let animationStarted = false;
  const typePreHeroText = () => {
    if (animationStarted) return;
    animationStarted = true;

    const preHeroTextEl = document.getElementById('pre-hero-text');
    preHeroTextEl.classList.add('typing');
    const text = "Hey! I'm still here!";
    let charIndex = 0;
    preHeroTextEl.textContent = '';

    const typingInterval = setInterval(() => {
      if (charIndex < text.length) {
        preHeroTextEl.textContent += text.charAt(charIndex);
        charIndex++;
      } else {
        clearInterval(typingInterval);
        preHeroTextEl.classList.remove('typing');
      }
    }, 100);
  };

  document.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
      event.preventDefault();
      typePreHeroText();
    }
  });

  // Theme Switcher
  const themeSwitch = document.getElementById('checkbox');
  const themeModeText = document.getElementById('theme-mode-text');
  const body = document.body;

  const setTheme = (isDark) => {
    if (isDark) {
      body.classList.add('dark-mode');
      themeSwitch.checked = true;
      themeModeText.textContent = 'Dark Mode';
    } else {
      body.classList.remove('dark-mode');
      themeSwitch.checked = false;
      themeModeText.textContent = 'Light Mode';
    }
  };

  // Check for saved theme preference
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    setTheme(savedTheme === 'dark');
  } else {
    // Check for system preference
    const prefersDark = window.matchMedia(
      '(prefers-color-scheme: dark)',
    ).matches;
    setTheme(prefersDark);
  }

  // Listen for toggle switch
  themeSwitch.addEventListener('change', () => {
    const isDark = themeSwitch.checked;
    setTheme(isDark);
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  });

  // Listen for system preference changes
  window
    .matchMedia('(prefers-color-scheme: dark)')
    .addEventListener('change', (e) => {
      setTheme(e.matches);
      localStorage.setItem('theme', e.matches ? 'dark' : 'light');
    });

  // Select all elements that need to animate on scroll
  const revealElements = document.querySelectorAll('.fade-up');

  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1,
  };

  const scrollObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');

        // Handle Voice Demo Animation
        if (
          entry.target.classList.contains('voice-demo-container') &&
          !entry.target.dataset.animated
        ) {
          entry.target.dataset.animated = true; // Mark as animated
          runVoiceDemoAnimation(entry.target);
        }
      }
    });
  }, observerOptions);

  revealElements.forEach((el) => {
    scrollObserver.observe(el);
  });

  // Voice Demo Animation Logic
  const runVoiceDemoAnimation = (container) => {
    const transcriptionEl = document.getElementById('transcription-text');
    const sentences = [
      'Where are my keys?',
      'Your keys were last seen on your desk.',
      ' ', // Clear screen
    ];
    let sentenceIndex = 0;

    const typeSentence = () => {
      if (sentenceIndex >= sentences.length) {
        // Reset and loop
        sentenceIndex = 0;
        setTimeout(() => {
          container.classList.remove('active');
          setTimeout(() => typeSentence(), 1000);
        }, 2000);
        return;
      }

      container.classList.add('active');
      const sentence = sentences[sentenceIndex];
      let charIndex = 0;
      transcriptionEl.textContent = '';

      const typingInterval = setInterval(() => {
        if (charIndex < sentence.length) {
          transcriptionEl.textContent += sentence.charAt(charIndex);
          charIndex++;
        } else {
          clearInterval(typingInterval);
          sentenceIndex++;
          setTimeout(typeSentence, 2000); // Wait before typing next sentence
        }
      }, 50);
    };

    setTimeout(typeSentence, 1000); // Initial delay
  };
});
