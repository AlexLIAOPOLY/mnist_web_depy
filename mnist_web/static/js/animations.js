/**
 * 动画效果控制脚本
 * 为MNIST Web应用提供平滑的页面过渡和交互动画
 */

// 页面加载动画
document.addEventListener('DOMContentLoaded', () => {
    // 添加页面过渡效果
    document.body.classList.add('page-loaded');
    
    // 为所有带animate-on-scroll类的元素添加滚动动画
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    
    // 初始化滚动检测
    initScrollAnimation(animatedElements);
    
    // 初始化交互动画
    initInteractionEffects();
});

// 滚动动画控制
function initScrollAnimation(elements) {
    // 创建IntersectionObserver实例
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // 元素进入视口，添加动画类
                entry.target.classList.add('animated');
                // 如果动画只需执行一次，则取消观察
                if (entry.target.dataset.animateOnce === 'true') {
                    observer.unobserve(entry.target);
                }
            } else if (entry.target.dataset.animateOnce !== 'true') {
                // 元素离开视口且需重复动画，移除动画类
                entry.target.classList.remove('animated');
            }
        });
    }, {
        threshold: 0.1 // 元素10%进入视口时触发
    });
    
    // 观察所有需要动画的元素
    elements.forEach(el => observer.observe(el));
}

// 交互动画效果
function initInteractionEffects() {
    // 按钮悬停效果
    const buttons = document.querySelectorAll('.button, button, .card, .feature-card');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', () => {
            button.classList.add('hover-active');
        });
        button.addEventListener('mouseleave', () => {
            button.classList.remove('hover-active');
        });
    });
    
    // 导航菜单项过渡效果
    const navItems = document.querySelectorAll('nav ul li');
    navItems.forEach((item, index) => {
        item.style.animationDelay = `${index * 0.1}s`;
        item.classList.add('nav-animate');
    });
    
    // 为标题添加打字机效果（如果存在特定类）
    const typingElements = document.querySelectorAll('.typing-effect');
    typingElements.forEach(element => {
        const text = element.textContent;
        element.textContent = '';
        typeText(element, text, 0, 50);
    });
}

// 打字机效果函数
function typeText(element, text, index, speed) {
    if (index < text.length) {
        element.textContent += text.charAt(index);
        index++;
        setTimeout(() => typeText(element, text, index, speed), speed);
    }
}

// 数据可视化动画（针对图表）
function animateChart(chart) {
    chart.data.datasets.forEach((dataset, i) => {
        dataset.animate = true;
        dataset.animationDuration = 1000;
        dataset.animationEasing = 'easeOutQuart';
    });
    chart.update();
}

// 页面过渡效果
function pageTransition(url) {
    document.body.classList.add('page-exit');
    setTimeout(() => {
        window.location.href = url;
    }, 300);
    return false;
}

// 导出一些公共函数以供其他脚本使用
window.MNIST = window.MNIST || {};
window.MNIST.animations = {
    animateChart,
    pageTransition
}; 