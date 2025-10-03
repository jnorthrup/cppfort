
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf050-switch-fallthrough/cf050-switch-fallthrough_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_switch_fallthroughi>:
100000360: 71000c1f    	cmp	w0, #0x3
100000364: 54000140    	b.eq	0x10000038c <__Z23test_switch_fallthroughi+0x2c>
100000368: 7100081f    	cmp	w0, #0x2
10000036c: 540000a0    	b.eq	0x100000380 <__Z23test_switch_fallthroughi+0x20>
100000370: 7100041f    	cmp	w0, #0x1
100000374: 540000a1    	b.ne	0x100000388 <__Z23test_switch_fallthroughi+0x28>
100000378: 528000c0    	mov	w0, #0x6                ; =6
10000037c: d65f03c0    	ret
100000380: 528000a0    	mov	w0, #0x5                ; =5
100000384: d65f03c0    	ret
100000388: 12800000    	mov	w0, #-0x1               ; =-1
10000038c: d65f03c0    	ret

0000000100000390 <_main>:
100000390: 528000c0    	mov	w0, #0x6                ; =6
100000394: d65f03c0    	ret
