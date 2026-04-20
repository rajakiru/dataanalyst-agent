export function Button({ children, className = '', variant = 'default', size = 'default', ...props }) {
  const base = 'inline-flex items-center justify-center font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:pointer-events-none disabled:opacity-50 rounded-lg'
  const variants = {
    default:   'bg-slate-900 text-white hover:bg-slate-800',
    primary:   'bg-indigo-600 text-white hover:bg-indigo-500',
    outline:   'border border-slate-200 bg-white text-slate-900 hover:bg-slate-50',
    ghost:     'text-slate-700 hover:bg-slate-100',
    secondary: 'bg-slate-100 text-slate-900 hover:bg-slate-200',
    danger:    'bg-red-600 text-white hover:bg-red-500',
  }
  const sizes = {
    default: 'h-9 px-4 py-2 text-sm',
    sm:      'h-8 px-3 text-xs',
    lg:      'h-11 px-6 text-base',
    icon:    'h-9 w-9',
  }
  return (
    <button
      className={`${base} ${variants[variant] ?? variants.default} ${sizes[size] ?? sizes.default} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}
