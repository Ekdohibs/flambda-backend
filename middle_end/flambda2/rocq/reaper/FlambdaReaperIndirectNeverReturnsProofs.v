(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* The proofs of FlambdaReaperIndirectNeverReturnsSpec.v. Nothing here
   needs review.

   The result with no source is that of
   FlambdaReaperDeadFramesProofs.dead_frame_of_call. The use: the call
   reads the code id off its witness, which has any usage; the witness
   reads the call witness of known arity of the callee, which may come
   from the known witness of the code dep, through the allocation of the
   closure ([closure_alloc]), as in
   FlambdaReaperDeadFramesProofs.marked_call. *)

From Stdlib Require Import List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperFlowSpec FlambdaReaperFlowProofs FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperDeadFramesSpec FlambdaReaperDeadFramesProofs
  FlambdaReaperIndirectNeverReturnsSpec.
From OxCamlFlambda2 Require FlambdaReaperNeverReturnsSpec.
Import ListNotations.
Import FlambdaReaperFlowProofs.WithLocal.

Theorem entered_used_never_returns : entered_used_never_returns_statement.
Proof.
  intros flag is_local u e T n s s1 ap callee id slots Hlocal Hslots Hfrag
    Hcheck Hrun Hexpr Hnu Hcallee Hclosure Hnr Hstep.
  destruct (dead_frame_of_call flag is_local u e T n s s1 ap callee id slots
              Hlocal Hslots Hfrag Hcheck Hrun Hexpr Hnu Hcallee Hclosure Hnr
              Hstep)
    as (cd & i & r & Hcd & Hr & Hrs).
  unfold FlambdaReaperNeverReturnsSpec.used_never_returns.
  apply andb_true_iff. split.
  { unfold code_never_returns. rewrite Hcd. apply existsb_exists.
    exists r. split; [ exact (nth_error_In _ _ Hr) | ].
    rewrite Hrs. reflexivity. }
  destruct (unit_inv flag is_local u e T n s Hlocal Hslots Hfrag Hcheck Hrun)
    as (noted & deps & Hc & Hconsts & Hsyms & _ & _ & _ & Hcds & _ & Hs).
  set (G := graph_of_facts (traverse_unit flag u e)) in *.
  set (sol := solution_of_tables T) in *.
  set (cds := unit_code_deps flag u e) in *.
  destruct s as [e0 vars K statics hp rd]. simpl in *. subst e0.
  destruct Hs as (ctx & ce & He & Hlive & _ & Hvars & _ & Hst & _ & _).
  destruct He as (_ & Hnoted_callee & _ & returns & exn & _ & _ & Hedges).
  apply known_call_edges_of in Hedges; [ | exact Hnu ].
  destruct Hedges as (witness & apply_node & called_node & _ & _ & _
                      & Hcalled & Hused & Hcond & Happly).
  destruct (Hcds id cd Hcd) as (_ & _ & _ & Hcodeid & Hasid).
  set (w := cd_known_arity_call_witness cd) in *.
  pose proof (simple_flows is_local G sol noted cds (unit_kept e) no_location
                Hc Hconsts Hsyms vars _ _ Hvars Hnoted_callee Hcallee)
    as Hfcallee.
  destruct (closure_alloc G sol cds (unit_kept e) no_location statics callee
              _ id slots Hst Hfcallee Hclosure)
    as (b & Hb & _ & Hcalled_b).
  unfold called_by, closure_dep_facts in Hcalled_b. simpl in Hcalled_b.
  rewrite Hcd in Hcalled_b. destruct Hcalled_b as (_ & Hcw & _).
  assert (Hw_self : may_come_from sol w w).
  { apply (alloc_self is_local G sol Hc _ _ _ Hcodeid).
    exact (as_hs is_local G sol Hc _ (any_source_input _ _ _ Hc _ Hasid)). }
  assert (Hw : may_come_from sol w witness).
  { apply (may_come_from_alias is_local G sol Hc _ _ apply_node);
      [ exact (cond_alias_alias is_local G sol Hc _ _ _ Hlive Hcond) | ].
    refine (may_come_from_read is_local G sol Hc w b _ _ _ _ _ Hcw Happly Hb
              Hw_self).
    apply (not_local is_local Hlocal); intros; discriminate. }
  refine (read_any_usage is_local G sol Hc w witness called_node _ _ _
            Hcodeid Hcalled Hw (any_usage_input _ _ _ Hc _ Hused)).
  apply (not_local is_local Hlocal); intros; discriminate.
Qed.
