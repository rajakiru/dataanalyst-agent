export function Badge({ children, className = '', variant = 'default', ...props }) {
  const variants = {
    default:   'bg-slate-900 text-white',
    secondary: 'bg-slate-100 text-slate-700',
    outline:   'border border-slate-200 text-slate-700 bg-white',
  }
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${variants[variant] ?? ''} ${className}`}
      {...props}
    >
      {children}
    </span>
  )
}
