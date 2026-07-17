document.addEventListener('DOMContentLoaded', () => {
    // Select all elements that need to animate on scroll
    const revealElements = document.querySelectorAll('.fade-up, .progress-bar, .hover-tilt');
    
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        // Lower threshold means it triggers slightly earlier for a smoother feel
        threshold: 0.1 
    };

    const scrollObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Add visible class to trigger CSS transitions
                entry.target.classList.add('visible');
            } else {
                // Remove visible class to replay animation when scrolling back up
                entry.target.classList.remove('visible');
            }
        });
    }, observerOptions);

    revealElements.forEach((el, index) => {
        // Stagger the animations slightly based on their order in the DOM
        if(el.classList.contains('hover-tilt')) {
             el.style.transitionDelay = `${(index % 3) * 0.1}s`;
        }
        scrollObserver.observe(el);
    });

    // Typing Effect for Subtitle
    const typeTarget = document.getElementById('type-text');
    if (typeTarget) {
        const text = "> Object memory for the ambient smart home_";
        let index = 0;
        
        function typeWriter() {
            if (index < text.length) {
                typeTarget.innerHTML += text.charAt(index);
                index++;
                setTimeout(typeWriter, 50);
            }
        }
        
        // Start typing after a short delay
        setTimeout(typeWriter, 500);
    }
    
    // Live Time Update for UI Mockup
    const timeTarget = document.getElementById('live-time');
    if(timeTarget) {
        setInterval(() => {
            const now = new Date();
            timeTarget.innerText = now.toISOString().split('T')[1].split('.')[0];
        }, 1000);
    }
});