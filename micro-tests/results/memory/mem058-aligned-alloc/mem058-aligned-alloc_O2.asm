
/Users/jim/work/cppfort/micro-tests/results/memory/mem058-aligned-alloc/mem058-aligned-alloc_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z18test_aligned_allocv>:
100000448: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
10000044c: 910003fd    	mov	x29, sp
100000450: 52800800    	mov	w0, #0x40               ; =64
100000454: 52800081    	mov	w1, #0x4                ; =4
100000458: 94000016    	bl	0x1000004b0 <_free+0x1000004b0>
10000045c: b40000a0    	cbz	x0, 0x100000470 <__Z18test_aligned_allocv+0x28>
100000460: 94000017    	bl	0x1000004bc <_free+0x1000004bc>
100000464: 52800540    	mov	w0, #0x2a               ; =42
100000468: a8c17bfd    	ldp	x29, x30, [sp], #0x10
10000046c: d65f03c0    	ret
100000470: 12800000    	mov	w0, #-0x1               ; =-1
100000474: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000478: d65f03c0    	ret

000000010000047c <_main>:
10000047c: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000480: 910003fd    	mov	x29, sp
100000484: 52800800    	mov	w0, #0x40               ; =64
100000488: 52800081    	mov	w1, #0x4                ; =4
10000048c: 94000009    	bl	0x1000004b0 <_free+0x1000004b0>
100000490: b40000a0    	cbz	x0, 0x1000004a4 <_main+0x28>
100000494: 9400000a    	bl	0x1000004bc <_free+0x1000004bc>
100000498: 52800540    	mov	w0, #0x2a               ; =42
10000049c: a8c17bfd    	ldp	x29, x30, [sp], #0x10
1000004a0: d65f03c0    	ret
1000004a4: 12800000    	mov	w0, #-0x1               ; =-1
1000004a8: a8c17bfd    	ldp	x29, x30, [sp], #0x10
1000004ac: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004b0 <__stubs>:
1000004b0: 90000030    	adrp	x16, 0x100004000 <_free+0x100004000>
1000004b4: f9400210    	ldr	x16, [x16]
1000004b8: d61f0200    	br	x16
1000004bc: 90000030    	adrp	x16, 0x100004000 <_free+0x100004000>
1000004c0: f9400610    	ldr	x16, [x16, #0x8]
1000004c4: d61f0200    	br	x16
