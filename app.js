'use strict';
const $ = (s, root=document) => root.querySelector(s);
let toastTimer;
function toast(message){const el=$('#toast');el.textContent=message;el.classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('visible'),3300);}
async function copyText(text){try{await navigator.clipboard.writeText(text);}catch{const field=document.createElement('textarea');field.value=text;field.style.position='fixed';field.style.opacity='0';document.body.append(field);field.select();const ok=document.execCommand('copy');field.remove();if(!ok)throw new Error('Copy unavailable');}}
$('#github-button')?.addEventListener('click',()=>toast('The GitHub repository link will be added here when available.'));
$('#copy-citation').addEventListener('click',async()=>{try{await copyText($('#citation-code').textContent);toast('BibTeX citation copied');}catch{toast('Please select the citation text to copy it.');}});
document.querySelectorAll('.copy-prompt').forEach(button=>button.addEventListener('click',async()=>{try{await copyText($('blockquote',button.closest('.demo-row')).textContent);toast('Prompt copied');}catch{toast('Please select the prompt text to copy it.');}}));
const demos=[...document.querySelectorAll('.demo-row')];
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-filter]').forEach(b=>{const active=b===button;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});let count=0;demos.forEach(row=>{const show=button.dataset.filter==='all'||row.dataset.category===button.dataset.filter;row.hidden=!show;if(show)count++;else $('video',row).pause();});$('#demo-count').textContent=count+(count===1?' scene':' scenes');}));
const demoObserver=new IntersectionObserver(entries=>entries.forEach(({target,isIntersecting})=>{if(isIntersecting&&!target.src){target.src=target.dataset.src;target.preload='metadata';target.load();}else if(!isIntersecting){target.pause();if(target.getAttribute('src')){target.removeAttribute('src');target.load();}}}),{rootMargin:'300px 0px'});
document.querySelectorAll('.demo-media video').forEach(video=>{video.style.backgroundImage=`url("${video.poster}")`;demoObserver.observe(video);video.addEventListener('play',()=>{document.querySelectorAll('video').forEach(other=>{if(other!==video)other.pause();});});video.addEventListener('error',()=>{const link=$('.video-fallback',video.closest('.demo-media'));link.textContent='Open video in a new tab ↗';});});
const overviewVideo=$('#overview-video');
overviewVideo.addEventListener('play',()=>document.querySelectorAll('.demo-media video').forEach(v=>v.pause()));
new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)overviewVideo.pause();},{threshold:0}).observe(overviewVideo);
const sections=['overview','motivation','method','evaluation','demo','citation'];const navLinks=[...document.querySelectorAll('.site-header nav a')];
let navFrame=false;
function updateNavigation(){let active='';sections.forEach(id=>{const rect=document.getElementById(id).getBoundingClientRect();if(rect.top<=160&&rect.bottom>160)active=id;});if(scrollY+innerHeight>=document.documentElement.scrollHeight-3)active='citation';navLinks.forEach(a=>{const on=a.hash==='#'+active;a.classList.toggle('active',on);if(on)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});navFrame=false;}
addEventListener('scroll',()=>{if(!navFrame){navFrame=true;requestAnimationFrame(updateNavigation);}}, {passive:true});
updateNavigation();

// Paper figures retain a direct link when scripting is unavailable.
const figureDialog=$('#figure-dialog'),figureFull=$('#figure-full');let figureScale=100;
function updateFigureScale(){figureFull.style.width=figureScale+'%';$('#figure-scale').value=figureScale+'%';$('#figure-less').disabled=figureScale<=100;$('#figure-more').disabled=figureScale>=300;}
const motionPreference=matchMedia('(prefers-reduced-motion: reduce)'),finePointer=matchMedia('(hover: hover) and (pointer: fine)');
document.querySelectorAll('.figure-zoom').forEach(link=>{
 link.addEventListener('click',event=>{if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;event.preventDefault();figureFull.src=link.href;figureFull.alt=$('img',link).alt;$('#figure-dialog-title').textContent=link.dataset.figureTitle;$('#figure-original').href=link.href;figureScale=100;updateFigureScale();figureDialog.showModal();document.body.classList.add('figure-is-open');$('.figure-viewport').scrollTo(0,0);});
 let frame;
 const reset=()=>{cancelAnimationFrame(frame);link.style.removeProperty('--figure-rx');link.style.removeProperty('--figure-ry');};
 link.addEventListener('pointermove',event=>{if(!finePointer.matches||motionPreference.matches||event.pointerType!=='mouse')return;cancelAnimationFrame(frame);const x=event.clientX,y=event.clientY;frame=requestAnimationFrame(()=>{const r=link.getBoundingClientRect();link.style.setProperty('--figure-rx',((.5-(y-r.top)/r.height)*1.6)+'deg');link.style.setProperty('--figure-ry',(((x-r.left)/r.width-.5)*1.6)+'deg');});});
 link.addEventListener('pointerleave',reset);motionPreference.addEventListener('change',reset);
});
$('#figure-more').addEventListener('click',()=>{figureScale=Math.min(300,figureScale+50);updateFigureScale();});
$('#figure-less').addEventListener('click',()=>{figureScale=Math.max(100,figureScale-50);updateFigureScale();});
$('#figure-close').addEventListener('click',()=>figureDialog.close());
figureDialog.addEventListener('click',event=>{if(event.target!==figureDialog)return;const r=figureDialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)figureDialog.close();});
figureDialog.addEventListener('close',()=>document.body.classList.remove('figure-is-open'));
