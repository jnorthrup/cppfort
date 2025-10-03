
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf093-recursive-fibonacci/cf093-recursive-fibonacci_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9fibonaccii>:
100000360: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000364: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000368: 910043fd    	add	x29, sp, #0x10
10000036c: 52800013    	mov	w19, #0x0               ; =0
100000370: 71000814    	subs	w20, w0, #0x2
100000374: 540000eb    	b.lt	0x100000390 <__Z9fibonaccii+0x30>
100000378: 51000400    	sub	w0, w0, #0x1
10000037c: 97fffff9    	bl	0x100000360 <__Z9fibonaccii>
100000380: 0b000273    	add	w19, w19, w0
100000384: aa1403e0    	mov	x0, x20
100000388: 71000814    	subs	w20, w0, #0x2
10000038c: 54ffff6a    	b.ge	0x100000378 <__Z9fibonaccii+0x18>
100000390: 0b130000    	add	w0, w0, w19
100000394: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000398: a8c24ff4    	ldp	x20, x19, [sp], #0x20
10000039c: d65f03c0    	ret

00000001000003a0 <_main>:
1000003a0: 52800140    	mov	w0, #0xa                ; =10
1000003a4: 17ffffef    	b	0x100000360 <__Z9fibonaccii>
