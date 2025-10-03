
/Users/jim/work/cppfort/micro-tests/results/memory/mem051-nothrow-new/mem051-nothrow-new_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z16test_nothrow_newv>:
100000448: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: 90000021    	adrp	x1, 0x100004000 <__ZnwmRKSt9nothrow_t+0x100004000>
100000458: f9400021    	ldr	x1, [x1]
10000045c: 52800080    	mov	w0, #0x4                ; =4
100000460: 94000023    	bl	0x1000004ec <__ZnwmRKSt9nothrow_t+0x1000004ec>
100000464: b4000060    	cbz	x0, 0x100000470 <__Z16test_nothrow_newv+0x28>
100000468: 52800548    	mov	w8, #0x2a               ; =42
10000046c: b9000008    	str	w8, [x0]
100000470: b4000080    	cbz	x0, 0x100000480 <__Z16test_nothrow_newv+0x38>
100000474: b9400013    	ldr	w19, [x0]
100000478: 9400001a    	bl	0x1000004e0 <__ZnwmRKSt9nothrow_t+0x1000004e0>
10000047c: 14000002    	b	0x100000484 <__Z16test_nothrow_newv+0x3c>
100000480: 12800013    	mov	w19, #-0x1              ; =-1
100000484: aa1303e0    	mov	x0, x19
100000488: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000048c: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000490: d65f03c0    	ret

0000000100000494 <_main>:
100000494: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000498: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000049c: 910043fd    	add	x29, sp, #0x10
1000004a0: 90000021    	adrp	x1, 0x100004000 <__ZnwmRKSt9nothrow_t+0x100004000>
1000004a4: f9400021    	ldr	x1, [x1]
1000004a8: 52800080    	mov	w0, #0x4                ; =4
1000004ac: 94000010    	bl	0x1000004ec <__ZnwmRKSt9nothrow_t+0x1000004ec>
1000004b0: b4000060    	cbz	x0, 0x1000004bc <_main+0x28>
1000004b4: 52800548    	mov	w8, #0x2a               ; =42
1000004b8: b9000008    	str	w8, [x0]
1000004bc: b4000080    	cbz	x0, 0x1000004cc <_main+0x38>
1000004c0: b9400013    	ldr	w19, [x0]
1000004c4: 94000007    	bl	0x1000004e0 <__ZnwmRKSt9nothrow_t+0x1000004e0>
1000004c8: 14000002    	b	0x1000004d0 <_main+0x3c>
1000004cc: 12800013    	mov	w19, #-0x1              ; =-1
1000004d0: aa1303e0    	mov	x0, x19
1000004d4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004d8: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000004dc: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004e0 <__stubs>:
1000004e0: 90000030    	adrp	x16, 0x100004000 <__ZnwmRKSt9nothrow_t+0x100004000>
1000004e4: f9400610    	ldr	x16, [x16, #0x8]
1000004e8: d61f0200    	br	x16
1000004ec: 90000030    	adrp	x16, 0x100004000 <__ZnwmRKSt9nothrow_t+0x100004000>
1000004f0: f9400a10    	ldr	x16, [x16, #0x10]
1000004f4: d61f0200    	br	x16
