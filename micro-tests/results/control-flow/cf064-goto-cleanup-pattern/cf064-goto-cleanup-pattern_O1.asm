
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf064-goto-cleanup-pattern/cf064-goto-cleanup-pattern_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z17test_goto_cleanupi>:
100000448: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: 37f801c0    	tbnz	w0, #0x1f, 0x10000048c <__Z17test_goto_cleanupi+0x44>
100000458: aa0003f3    	mov	x19, x0
10000045c: 52800080    	mov	w0, #0x4                ; =4
100000460: 94000014    	bl	0x1000004b0 <__Znwm+0x1000004b0>
100000464: 52800548    	mov	w8, #0x2a               ; =42
100000468: b9000008    	str	w8, [x0]
10000046c: 7100027f    	cmp	w19, #0x0
100000470: 5a9f1113    	csinv	w19, w8, wzr, ne
100000474: b4000040    	cbz	x0, 0x10000047c <__Z17test_goto_cleanupi+0x34>
100000478: 9400000b    	bl	0x1000004a4 <__Znwm+0x1000004a4>
10000047c: aa1303e0    	mov	x0, x19
100000480: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000484: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000488: d65f03c0    	ret
10000048c: d2800000    	mov	x0, #0x0                ; =0
100000490: 12800013    	mov	w19, #-0x1              ; =-1
100000494: b5ffff20    	cbnz	x0, 0x100000478 <__Z17test_goto_cleanupi+0x30>
100000498: 17fffff9    	b	0x10000047c <__Z17test_goto_cleanupi+0x34>

000000010000049c <_main>:
10000049c: 52800540    	mov	w0, #0x2a               ; =42
1000004a0: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004a4 <__stubs>:
1000004a4: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000004a8: f9400210    	ldr	x16, [x16]
1000004ac: d61f0200    	br	x16
1000004b0: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000004b4: f9400610    	ldr	x16, [x16, #0x8]
1000004b8: d61f0200    	br	x16
