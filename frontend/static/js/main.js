// Minimal client-side navigation to static pages to preserve flow
(function(){
  const form = document.getElementById('signin-form');
  if (form) {
    form.addEventListener('submit', function(e){
      e.preventDefault();
      window.location.href = 'homepage.html';
    });
  }
})();

