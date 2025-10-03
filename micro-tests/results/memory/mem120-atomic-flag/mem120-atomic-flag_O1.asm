
/Users/jim/work/cppfort/micro-tests/results/memory/mem120-atomic-flag/mem120-atomic-flag_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z16test_atomic_flagv>:
1000003f8: 90000028    	adrp	x8, 0x100004000 <_flag>
1000003fc: 91000108    	add	x8, x8, #0x0
100000400: 52800029    	mov	w9, #0x1                ; =1
100000404: 38e98109    	swpalb	w9, w9, [x8]
100000408: 089ffd1f    	stlrb	wzr, [x8]
10000040c: 12000120    	and	w0, w9, #0x1
100000410: d65f03c0    	ret

0000000100000414 <_main>:
100000414: 90000028    	adrp	x8, 0x100004000 <_flag>
100000418: 91000108    	add	x8, x8, #0x0
10000041c: 52800029    	mov	w9, #0x1                ; =1
100000420: 38e98109    	swpalb	w9, w9, [x8]
100000424: 089ffd1f    	stlrb	wzr, [x8]
100000428: 12000120    	and	w0, w9, #0x1
10000042c: d65f03c0    	ret
