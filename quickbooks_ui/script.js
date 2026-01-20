document.addEventListener('DOMContentLoaded', () => {
    // Config Check
    fetch('qb_config.json')
        .then(response => response.json())
        .then(config => {
            if (config.qb_mode) {
                console.log("QuickBooks Mode active");
                document.body.classList.add('qb-mode-active');
            }
        })
        .catch(err => console.error("Error loading config:", err));

    // Elements
    const navCreateBtn = document.getElementById('navCreateBtn');
    const modalOverlay = document.getElementById('accountModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const form = document.getElementById('accountForm');

    // Toggle Modal
    function openModal() {
        modalOverlay.classList.add('open');
        document.getElementById('businessName').focus();
    }

    function closeModal() {
        modalOverlay.classList.remove('open');
        form.reset();
        clearSuggestions();
    }

    navCreateBtn.addEventListener('click', openModal);
    closeModalBtn.addEventListener('click', closeModal);
    cancelBtn.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) closeModal();
    });

    // Autopopulation Data
    const mockDb = {
        industries: [
            "Technology Consulting",
            "Software Development",
            "Retail Clothing",
            "Restaurant & Food Service",
            "Real Estate Agency",
            "Construction & Trades",
            "Graphic Design",
            "Health & Wellness"
        ],
        companies: [
            { name: "Acme Corp", industry: "Technology Consulting", type: "corp", employees: "50+" },
            { name: "Beta Bakery", industry: "Restaurant & Food Service", type: "sole", employees: "2-9" },
            { name: "Gamma Graphics", industry: "Graphic Design", type: "llc", employees: "1" }
        ]
    };

    // Autocomplete Logic
    const businessNameInput = document.getElementById('businessName');
    const industryInput = document.getElementById('industry');
    const busSuggestions = document.getElementById('businessSuggestions');
    const indSuggestions = document.getElementById('industrySuggestions');

    function showSuggestions(input, suggestionsContainer, items, onSelect) {
        suggestionsContainer.innerHTML = '';
        if (items.length > 0) {
            suggestionsContainer.classList.add('visible');
            items.forEach(item => {
                const div = document.createElement('div');
                div.classList.add('suggestion-item');
                div.textContent = typeof item === 'object' ? item.name : item;
                div.addEventListener('click', () => {
                    input.value = typeof item === 'object' ? item.name : item;
                    suggestionsContainer.classList.remove('visible');
                    if (onSelect) onSelect(item);
                });
                suggestionsContainer.appendChild(div);
            });
        } else {
            suggestionsContainer.classList.remove('visible');
        }
    }

    function clearSuggestions() {
        busSuggestions.classList.remove('visible');
        indSuggestions.classList.remove('visible');
    }

    // Business Name Input Listener
    businessNameInput.addEventListener('input', (e) => {
        const val = e.target.value.toLowerCase();
        if (val.length < 1) {
            clearSuggestions();
            return;
        }

        const matches = mockDb.companies.filter(c => c.name.toLowerCase().includes(val));
        showSuggestions(businessNameInput, busSuggestions, matches, (selectedCompany) => {
            // Auto-populate other fields
            industryInput.value = selectedCompany.industry;
            document.getElementById('businessType').value = selectedCompany.type;
            document.getElementById('employeeCount').value = selectedCompany.employees;

            // Visual feedback for autopopulation
            industryInput.style.backgroundColor = "#e8f5e9";
            setTimeout(() => industryInput.style.backgroundColor = "", 1000);
        });
    });

    // Industry Input Listener
    industryInput.addEventListener('input', (e) => {
        const val = e.target.value.toLowerCase();
        if (val.length < 1) {
            indSuggestions.classList.remove('visible');
            return;
        }

        const matches = mockDb.industries.filter(i => i.toLowerCase().includes(val));
        showSuggestions(industryInput, indSuggestions, matches);
    });

    // Close suggestions on click outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.form-group')) {
            clearSuggestions();
        }
    });
});
