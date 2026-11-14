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

(* Lemmas for the proofs of the statements of FlambdaRegionsCheckedSpec.v:
   the step of a [Let] and its check, each as one function of the defining
   expression, so that a proof by cases on the defining expression is made
   once. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaRegionsSpec FlambdaRegionsCheckSpec.
Import ListNotations.

Definition region_named (d : named) : bool :=
  match d with
  | Prim (Region_primitive _) _ => true
  | _ => false
  end.

(* The step of a [Let] of a named expression that is not a primitive of
   regions, the test of the simples done. *)
Definition let_step (rs : region_state) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) : region_step_result :=
  match marker_of x body with
  | Marker y region body' => local_allocation rs x y kind d region body'
  | Bad_marker => Region_undefined Not_of_the_conversion
  | No_marker =>
    if is_allocation d && existsb (simple_local rs) (simples_of_named d)
    then Region_undefined Heap_holds_local
    else plain_step rs
  end.

Lemma region_step_let (rs : region_state) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  state_expr (rs_state rs) = Let x kind d body ->
  region_named d = false ->
  simples_fault rs (evaluated_simples (state_expr (rs_state rs)))
    (region_places rs)
  = None ->
  region_step rs = let_step (count_uses rs) x kind d body.
Proof.
  intros He Hd Hfault. unfold region_step. cbv zeta. rewrite Hfault, He.
  destruct d as [s | [u | b | v | hp | rp] args | | | | | ];
    first [ reflexivity | discriminate ].
Qed.

(* The step of a [Let] of a primitive of regions. *)
Definition region_let_step (rs : region_state) (x : variable)
    (kind : flambda_kind) (p : region_primitive) (args : list simple)
    (body : expr) : region_step_result :=
  match p, args with
  | Begin_region, [] => begin_region rs x kind Plain_mark body
  | Begin_try_region, [] => begin_region rs x kind Try_mark body
  | End_region, [region] => end_region rs x kind Plain_mark region body
  | End_try_region, [region] => end_region rs x kind Try_mark region body
  | _, _ => Region_undefined Not_of_the_conversion
  end.

Lemma region_step_region_let (rs : region_state) (x : variable)
    (kind : flambda_kind) (p : region_primitive) (args : list simple)
    (body : expr) :
  state_expr (rs_state rs) = Let x kind (Prim (Region_primitive p) args) body ->
  region_step rs = region_let_step (count_uses rs) x kind p args body.
Proof.
  intros He. unfold region_step. cbv zeta. rewrite He.
  cbn [evaluated_simples simples_fault]. reflexivity.
Qed.

Section Check.

Variable cert : certificate.
Variable codes : list (code_id * code).

(* The check of such a [Let]. *)
Definition check_let (cs : check_state) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) : bool :=
  let simples := simples_of_named d in
  not_a_region cs x
  && forallb (check_usable cs) simples
  && match marker_of x body with
     | Marker y region _ =>
       is_allocation d
       && kind_eqb kind Value
       && not_a_region cs y
       && match region with
          | None => true
          | Some (Var r) => among r (cs_open cs)
          | Some (Symbol _ | Const _) => false
          end
       && forallb (closure_may_be_local cert) (closure_code d)
       && match body with
          | Let _ _ _ body' =>
            check_expr cert codes
              (bind_lifetime y (newest_lifetime cs)
                 (bind_lifetime x (newest_lifetime cs) cs))
              body'
          | _ => false
          end
     | Bad_marker => false
     | No_marker =>
       if is_allocation d
       then
         forallb (fun s => is_forever (lifetime_of cs s)) simples
         && check_expr cert codes (bind_lifetime x Forever cs) body
       else
         check_expr cert codes (bind_lifetime x (named_lifetime cs d) cs) body
     end.

Lemma check_expr_let (cs : check_state) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  region_named d = false ->
  check_expr cert codes cs (Let x kind d body) = check_let cs x kind d body.
Proof.
  destruct d as [s | [u | b | v | hp | rp] args | | | | | ];
    first [ reflexivity | discriminate ].
Qed.

(* The check of a [Let] of a primitive of regions. *)
Definition check_region_let (cs : check_state) (x : variable)
    (kind : flambda_kind) (p : region_primitive) (args : list simple)
    (body : expr) : bool :=
  match p, args with
  | (Begin_region | Begin_try_region), [] =>
    kind_eqb kind Region
    && not_a_region cs x
    && check_expr cert codes
         (Check_state ((x, Gone) :: cs_env cs) (x :: cs_open cs)
            (x :: cs_regions cs) (cs_conts cs))
         body
  | (End_region | End_try_region), [Var r] =>
    kind_eqb kind Value
    && not_a_region cs x
    && among r (cs_open cs)
    && check_expr cert codes
         (Check_state ((x, Forever) :: cs_env cs)
            (regions_below r (cs_open cs)) (cs_regions cs) (cs_conts cs))
         body
  | _, _ => false
  end.

Lemma check_expr_region_let (cs : check_state) (x : variable)
    (kind : flambda_kind) (p : region_primitive) (args : list simple)
    (body : expr) :
  check_expr cert codes cs (Let x kind (Prim (Region_primitive p) args) body)
  = check_region_let cs x kind p args body.
Proof. reflexivity. Qed.

End Check.
