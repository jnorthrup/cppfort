
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf093-recursive-fibonacci/cf093-recursive-fibonacci_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9fibonaccii>:
100000360: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000364: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000368: 910043fd    	add	x29, sp, #0x10
10000036c: 7100081f    	cmp	w0, #0x2
100000370: 540000ca    	b.ge	0x100000388 <__Z9fibonaccii+0x28>
100000374: 52800013    	mov	w19, #0x0               ; =0
100000378: 0b130000    	add	w0, w0, w19
10000037c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000380: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000384: d65f03c0    	ret
100000388: 52800013    	mov	w19, #0x0               ; =0
10000038c: aa0003f4    	mov	x20, x0
100000390: 51000680    	sub	w0, w20, #0x1
100000394: 97fffff3    	bl	0x100000360 <__Z9fibonaccii>
100000398: aa0003e8    	mov	x8, x0
10000039c: 51000a80    	sub	w0, w20, #0x2
1000003a0: 0b130113    	add	w19, w8, w19
1000003a4: 71000e9f    	cmp	w20, #0x3
1000003a8: aa0003f4    	mov	x20, x0
1000003ac: 54ffff28    	b.hi	0x100000390 <__Z9fibonaccii+0x30>
1000003b0: 0b130000    	add	w0, w0, w19
1000003b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b8: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000003bc: d65f03c0    	ret

00000001000003c0 <_main>:
1000003c0: 52800140    	mov	w0, #0xa                ; =10
1000003c4: 17ffffe7    	b	0x100000360 <__Z9fibonaccii>
